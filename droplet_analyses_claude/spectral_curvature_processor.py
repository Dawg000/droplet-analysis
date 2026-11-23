"""
spectral_curvature_processor.py
================================
Top-level batch processor: parquet → HDF5 via the spectral Gaussian pipeline.

This module owns the ``ProcessPoolExecutor`` worker loop and CLI.  It is
deliberately thin: all physics lives in ``spectral_pipeline.py``, all I/O in
``contour_io.py`` and ``curvature_hdf5_writer.py``.  Adding a SavGol pipeline
later means writing a parallel worker function and a new ``main()`` entry
point, reusing the same HDF5 writer and contour I/O infrastructure.

Worker architecture
-------------------
Each worker subprocess:
  1. Loads its trajectory's rows from parquet (filter by trajectory_id).
  2. Decodes RLE masks → raw contours (CPU-only; pycocotools is not GPU-aware).
  3. Computes arc-length-resampled contours, Gaussian-windowed curvature,
     and Fourier modes per frame — all on CPU via ``spectral_pipeline``.
  4. Optionally writes a contour cache.
  5. Returns a minimal dict; the main process writes to HDF5 (avoids
     concurrent HDF5 writes from worker processes).

GPU note: CuPy GPU contexts cannot cross process boundaries.  The batch FFT
GPU path in ``spectral_pipeline.process_trajectory_frames`` is called inside
the worker where the GPU context is fresh.  The RTX 5090's overhead for
initializing a CUDA context per-worker is non-trivial; if you find workers
spending disproportionate time on first-frame GPU init, set
``use_gpu_in_workers=False`` and benchmark both configurations.

CLI usage
---------
::

    python spectral_curvature_processor.py \\
        /path/to/masks.parquet \\
        [--output /path/to/output.h5] \\
        [--lambda-cut 16.0] \\
        [--n-resample 64] \\
        [--pixel-size 2.46] \\
        [--frame-rate 800.0] \\
        [--workers 12] \\
        [--no-resume] \\
        [--cache-dir /path/to/contour_cache] \\
        [--write-cache] \\
        [--no-gpu]

Default output: ``<parquet_parent>/spectral_curvature_processor/<stem>_pv<version>.h5``
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import contour_io
import curvature_hdf5_writer as hdf5_writer
import spectral_pipeline as sp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RLE column discovery
# ---------------------------------------------------------------------------

_RLE_COLUMN_CANDIDATES = ["rle_counts_str", "rle_mask", "rle", "mask"]
_CENTROID_X_CANDIDATES = ["centroid_x", "cx"]
_CENTROID_Y_CANDIDATES = ["centroid_y", "cy"]
_PERIMETER_CANDIDATES = ["perimeter"]
_AREA_CANDIDATES = ["area"]


def _find_column(row_or_cols, candidates: list[str]) -> str | None:
    """Return the first candidate column name present in *row_or_cols*."""
    cols = (
        row_or_cols if isinstance(row_or_cols, (list, set)) else list(row_or_cols.index)
    )
    for c in candidates:
        if c in cols:
            return c
    return None


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------


def _spectral_worker(args: tuple) -> tuple[int, dict | None, str | None]:
    """Process one trajectory in a worker subprocess.

    Parameters
    ----------
    args : (parquet_path, trajectory_id, params_dict, cache_dir, write_cache, use_gpu)

    Returns
    -------
    (trajectory_id, record_dict | None, error_message | None)

    ``record_dict`` contains all arrays needed by ``hdf5_writer.write_trajectory``.
    """
    (
        parquet_path,
        trajectory_id,
        params_dict,
        cache_dir,  # str | None  — directory to look for / write cache
        write_cache,  # bool
        use_gpu,  # bool
    ) = args

    try:
        params = sp.SpectralPipelineParams(
            **{
                k: v
                for k, v in params_dict.items()
                if k in sp.SpectralPipelineParams.__dataclass_fields__
            }
        )

        # ── Load parquet rows ────────────────────────────────────────────────
        try:
            df = pd.read_parquet(
                parquet_path,
                filters=[("trajectory_id", "=", trajectory_id)],
            )
        except (ValueError, TypeError):
            df_full = pd.read_parquet(parquet_path)
            df = df_full[df_full["trajectory_id"] == trajectory_id].copy()

        if len(df) == 0:
            return trajectory_id, None, "No rows for trajectory in parquet"

        df = df.sort_values("frame").reset_index(drop=True)

        # Discover columns once
        sample_cols = list(df.columns)
        rle_col = _find_column(sample_cols, _RLE_COLUMN_CANDIDATES)
        cx_col = _find_column(sample_cols, _CENTROID_X_CANDIDATES)
        cy_col = _find_column(sample_cols, _CENTROID_Y_CANDIDATES)
        peri_col = _find_column(sample_cols, _PERIMETER_CANDIDATES)
        area_col = _find_column(sample_cols, _AREA_CANDIDATES)

        if rle_col is None:
            return trajectory_id, None, f"No RLE column found. Columns: {sample_cols}"

        # ── Decode contours ──────────────────────────────────────────────────
        # Check cache first if a cache_dir is provided
        loaded_from_cache = False
        frame_nums_cache: list[int] = []
        contours_cache: list[np.ndarray] = []

        if cache_dir is not None and contour_io.cache_exists(cache_dir, trajectory_id):
            try:
                frame_arr, contours_cache = contour_io.load_all_contours_from_cache(
                    cache_dir, trajectory_id
                )
                frame_nums_cache = frame_arr.tolist()
                loaded_from_cache = True
            except Exception as exc:
                logger.warning(
                    f"Cache load failed for trajectory {trajectory_id}: {exc}. "
                    "Falling back to parquet decode."
                )

        frame_nums: list[int] = []
        raw_contours: list[np.ndarray] = []
        centroids: list[list[float]] = []
        orig_perimeters: list[float] = []
        orig_areas: list[float] = []
        first_error: str | None = None

        if loaded_from_cache:
            # Use cache; still need to pull centroids/perimeters/areas from parquet
            frame_set = set(frame_nums_cache)
            for _, row in df.iterrows():
                fn = int(row["frame"])
                if fn not in frame_set:
                    continue
                idx = frame_nums_cache.index(fn)
                frame_nums.append(fn)
                raw_contours.append(contours_cache[idx])
                centroids.append(
                    [
                        float(row[cx_col]) if cx_col else 0.0,
                        float(row[cy_col]) if cy_col else 0.0,
                    ]
                )
                orig_perimeters.append(float(row[peri_col]) if peri_col else 0.0)
                orig_areas.append(float(row[area_col]) if area_col else 0.0)
        else:
            # Decode from RLE
            for _, row in df.iterrows():
                try:
                    contour = contour_io.get_contour_from_rle(
                        row[rle_col],
                        int(row["frame"]),
                        int(row.get("object_id", 0)),
                        params.image_height,
                        params.image_width,
                    )
                    frame_nums.append(int(row["frame"]))
                    raw_contours.append(contour)
                    centroids.append(
                        [
                            float(row[cx_col]) if cx_col else 0.0,
                            float(row[cy_col]) if cy_col else 0.0,
                        ]
                    )
                    orig_perimeters.append(float(row[peri_col]) if peri_col else 0.0)
                    orig_areas.append(float(row[area_col]) if area_col else 0.0)

                except Exception as exc:
                    if first_error is None:
                        first_error = (
                            f"Frame {row['frame']}: {type(exc).__name__}: {exc}"
                        )
                    continue

        if len(raw_contours) == 0:
            msg = "No frames decoded"
            if first_error:
                msg += f" (first error: {first_error})"
            return trajectory_id, None, msg

        # ── Write cache if requested ─────────────────────────────────────────
        if write_cache and cache_dir is not None and not loaded_from_cache:
            try:
                contour_io.write_trajectory_cache(
                    cache_dir, trajectory_id, frame_nums, raw_contours
                )
            except Exception as exc:
                logger.warning(
                    f"Cache write failed for trajectory {trajectory_id}: {exc}"
                )

        # ── Compute equivalent radii and areas ───────────────────────────────
        R_um_list: list[float] = []
        areas_um2_list: list[float] = []

        for contour in raw_contours:
            area_px2 = cv2.contourArea(contour.reshape(-1, 1, 2).astype(np.float32))
            area_um2 = area_px2 * params.pixel_size**2
            R_um = float(np.sqrt(area_px2 / np.pi)) * params.pixel_size
            R_um_list.append(R_um)
            areas_um2_list.append(area_um2)

        # ── Shape orientation (from raw contour, independent of the spectral
        # pipeline's arc-length parameterization — see contour_io.contour_orientation
        # for why this, not the mode-2 Fourier phase, is the field to use for
        # any comparison to a lab-frame axis) ─────────────────────────────────
        moment_angles: list[float] = []
        ellipse_angles: list[float] = []
        ellipse_aspect_ratios: list[float] = []
        for contour in raw_contours:
            orient = contour_io.contour_orientation(contour)
            moment_angles.append(orient["moment_angle_rad"])
            ellipse_angles.append(orient["ellipse_angle_rad"])
            ellipse_aspect_ratios.append(orient["ellipse_aspect_ratio"])

        # ── Spectral pipeline ────────────────────────────────────────────────
        result: sp.TrajectoryResult = sp.process_trajectory_frames(
            contours=raw_contours,
            frame_nums=frame_nums,
            R_um_per_frame=R_um_list,
            areas_um2_per_frame=areas_um2_list,
            params=params,
            use_gpu=use_gpu,
        )

        record = {
            "frames": result.frames,
            "curvature_matrix": result.curvature_matrix,
            "fourier_amplitudes": result.fourier_amplitudes,
            "fourier_phases": result.fourier_phases,
            "equivalent_radii": result.equivalent_radii_um,
            "perimeters": result.perimeters_um,
            "areas": result.areas_um2,
            "m_max_per_frame": result.m_max_per_frame,
            "gaussian_sigmas": result.gaussian_sigmas,
            "smoothed_equivalent_radii": result.smoothed_equivalent_radii_um,
            "smoothed_perimeters": result.smoothed_perimeters_um,
            "turning_numbers": result.turning_numbers,
            "centroids": np.array(centroids, dtype=np.float32),
            "original_perimeters": np.array(orig_perimeters, dtype=np.float32),
            "original_areas": np.array(orig_areas, dtype=np.float32),
            "moment_angle_rad": np.array(moment_angles, dtype=np.float32),
            "ellipse_angle_rad": np.array(ellipse_angles, dtype=np.float32),
            "ellipse_aspect_ratio": np.array(ellipse_aspect_ratios, dtype=np.float32),
        }

        return trajectory_id, record, None

    except Exception as exc:
        import traceback

        return (
            trajectory_id,
            None,
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------


class SpectralCurvatureProcessor:
    """Batch processor: parquet → curvature HDF5 via spectral Gaussian pipeline.

    Parameters
    ----------
    params       : SpectralPipelineParams controlling the physics.
    cache_dir    : directory for contour cache (None = no cache I/O).
    write_cache  : if True AND cache_dir is set, write a .npz cache per trajectory.
    use_gpu      : allow GPU FFT in workers (requires CuPy in each worker process).
    max_workers  : number of parallel worker processes (None = min(cpu_count, 12)).
    """

    def __init__(
        self,
        params: sp.SpectralPipelineParams,
        cache_dir: str | None = None,
        write_cache: bool = False,
        use_gpu: bool = True,
        max_workers: int | None = None,
    ):
        self.params = params
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.write_cache = write_cache
        self.use_gpu = use_gpu
        self.max_workers = max_workers or min(os.cpu_count() or 4, 12)

        if write_cache and cache_dir is None:
            raise ValueError("write_cache=True requires cache_dir to be set")

    def process_parquet(
        self,
        parquet_path: str,
        output_path: str,
        trajectory_ids: list[int] | None = None,
        parallel: bool = True,
        resume: bool = True,
    ) -> dict:
        """Run the pipeline on all (or selected) trajectories in *parquet_path*.

        Parameters
        ----------
        parquet_path   : input parquet file with RLE masks.
        output_path    : destination HDF5 file.
        trajectory_ids : explicit list of IDs to process (None = all).
        parallel       : use ProcessPoolExecutor if True.
        resume         : skip trajectories already in the output file.

        Returns
        -------
        summary dict with keys: status, n_processed, n_failed, elapsed_time, …
        """
        start_time = time.time()
        parquet_path = str(parquet_path)
        output_path = str(output_path)

        # ── Discover trajectory IDs ──────────────────────────────────────────
        if trajectory_ids is None:
            logger.info(f"Reading trajectory IDs from {parquet_path}")
            df_ids = pd.read_parquet(parquet_path, columns=["trajectory_id"])
            trajectory_ids = df_ids["trajectory_id"].unique().tolist()
            logger.info(f"Found {len(trajectory_ids)} trajectories")

        # ── Resume: skip already-processed ──────────────────────────────────
        if resume:
            done = hdf5_writer.already_processed_ids(output_path)
            if done:
                logger.info(f"Resuming: {len(done)} trajectories already in output")
            trajectory_ids = [t for t in trajectory_ids if t not in done]

        if not trajectory_ids:
            logger.info("Nothing to process")
            return {"status": "complete", "n_processed": 0, "n_failed": 0}

        logger.info(
            f"Processing {len(trajectory_ids)} trajectories "
            f"({'parallel' if parallel else 'sequential'}, "
            f"{self.max_workers} workers)"
        )

        # ── Try to find cache dir automatically if not provided ──────────────
        effective_cache_dir = self.cache_dir
        if effective_cache_dir is None:
            found = contour_io.find_cache_dir(Path(parquet_path).parent)
            if found:
                logger.info(f"Auto-detected contour cache: {found}")
                effective_cache_dir = str(found)

        # ── Build param dict for workers (serializable) ──────────────────────
        params_dict = self.params.to_dict()
        # Also pass raw dataclass fields for reconstruction in worker
        params_dict.update(
            {k: getattr(self.params, k) for k in self.params.__dataclass_fields__}
        )

        worker_args = [
            (
                parquet_path,
                tid,
                params_dict,
                effective_cache_dir,
                self.write_cache,
                self.use_gpu,
            )
            for tid in trajectory_ids
        ]

        successful: list[int] = []
        failed: list[tuple[int, str]] = []

        # ── Open HDF5 once in the main process ──────────────────────────────
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        h5 = h5py.File(output_path, "a")
        hdf5_writer._ensure_metadata(h5, params_dict)

        try:
            if parallel and len(trajectory_ids) > 1:
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        executor.submit(_spectral_worker, arg): arg[1]
                        for arg in worker_args
                    }
                    with tqdm(total=len(trajectory_ids), desc="Trajectories") as pbar:
                        for future in as_completed(futures):
                            tid, record, error = future.result()
                            self._handle_result(
                                h5, tid, record, error, successful, failed
                            )
                            pbar.update(1)
                            if successful:
                                elapsed = time.time() - start_time
                                rate = len(successful) / elapsed
                                pbar.set_description(f"Trajectories ({rate:.1f}/s)")
            else:
                for arg in tqdm(worker_args, desc="Trajectories"):
                    tid, record, error = _spectral_worker(arg)
                    self._handle_result(h5, tid, record, error, successful, failed)

        finally:
            h5.close()

        elapsed = time.time() - start_time

        if failed:
            n_show = min(3, len(failed))
            logger.error(f"First {n_show} failures:")
            for tid, msg in failed[:n_show]:
                logger.error(f"  trajectory_{tid}: {msg[:300]}")

        summary = {
            "status": "complete",
            "n_processed": len(successful),
            "n_failed": len(failed),
            "successful_ids": successful,
            "failed_ids": [t for t, _ in failed],
            "elapsed_time": elapsed,
            "trajectories_per_second": len(successful) / elapsed
            if elapsed > 0
            else 0.0,
        }

        logger.info(
            f"Done: {len(successful)} processed, {len(failed)} failed "
            f"in {elapsed:.1f}s ({summary['trajectories_per_second']:.2f} traj/s)"
        )
        return summary

    def _handle_result(
        self,
        h5: h5py.File,
        tid: int,
        record: dict | None,
        error: str | None,
        successful: list,
        failed: list,
    ) -> None:
        if record is not None:
            hdf5_writer.write_trajectory(
                h5, tid, record, frame_rate=self.params.frame_rate
            )
            successful.append(tid)
        else:
            hdf5_writer.record_failure(h5, tid, error or "unknown error")
            failed.append((tid, error or "unknown error"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_default_output(parquet_path: str) -> str:
    p = Path(parquet_path)
    out_dir = p.parent / "spectral_curvature_processor_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Version-tagged so a v2 run never lands in (or is refused by) a v1 file
    return str(out_dir / f"{p.stem}_pv{sp.PIPELINE_VERSION}.h5")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    ap = argparse.ArgumentParser(
        description="Spectral Gaussian curvature pipeline: parquet → HDF5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("parquet", help="Input parquet file with RLE masks")
    ap.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output HDF5 path. Default: <parquet_parent>/spectral_curvature_processor/<stem>_pv<version>.h5",
    )
    ap.add_argument(
        "--lambda-cut",
        type=float,
        default=16.0,
        help="Gaussian −3 dB cutoff wavelength (µm)",
    )
    ap.add_argument(
        "--n-resample",
        type=int,
        default=64,
        help="Number of arc-length-resampled contour points (must be even)",
    )
    ap.add_argument(
        "--pixel-size", type=float, default=2.46, help="Pixel pitch (µm/pixel)"
    )
    ap.add_argument(
        "--frame-rate", type=float, default=800.0, help="Acquisition frame rate (Hz)"
    )
    ap.add_argument("--image-height", type=int, default=800)
    ap.add_argument("--image-width", type=int, default=1280)
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: min(cpu_count, 12))",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-process trajectories already in the output file",
    )
    ap.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (useful for debugging)",
    )
    ap.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration even if CuPy is available",
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="Contour cache directory. If omitted the script searches "
        "automatically around the parquet location.",
    )
    ap.add_argument(
        "--write-cache",
        action="store_true",
        help="Write contour .npz cache files (requires --cache-dir)",
    )
    ap.add_argument(
        "--trajectory-ids",
        nargs="+",
        type=int,
        default=None,
        help="Process only these trajectory IDs (space-separated)",
    )

    args = ap.parse_args()

    if args.write_cache and args.cache_dir is None:
        ap.error("--write-cache requires --cache-dir")

    params = sp.SpectralPipelineParams(
        pixel_size=args.pixel_size,
        frame_rate=args.frame_rate,
        n_resample=args.n_resample,
        lambda_cut_um=args.lambda_cut,
        image_height=args.image_height,
        image_width=args.image_width,
    )

    output_path = args.output or _build_default_output(args.parquet)
    logger.info(f"Output: {output_path}")

    processor = SpectralCurvatureProcessor(
        params=params,
        cache_dir=args.cache_dir,
        write_cache=args.write_cache,
        use_gpu=not args.no_gpu,
        max_workers=args.workers,
    )

    summary = processor.process_parquet(
        parquet_path=args.parquet,
        output_path=output_path,
        trajectory_ids=args.trajectory_ids,
        parallel=not args.no_parallel,
        resume=not args.no_resume,
    )

    print(f"\n{'─' * 60}")
    print(f"  Processed : {summary['n_processed']}")
    print(f"  Failed    : {summary['n_failed']}")
    print(f"  Wall time : {summary['elapsed_time']:.1f} s")
    print(f"  Rate      : {summary['trajectories_per_second']:.2f} traj/s")
    print(f"  Output    : {output_path}")


if __name__ == "__main__":
    main()
