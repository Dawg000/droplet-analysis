"""
curvature_hdf5_writer.py
========================
HDF5 I/O layer for curvature analysis results.

This module is intentionally **pipeline-agnostic**: it knows the on-disk
schema but nothing about how curvatures were computed.  Any pipeline
(spectral Gaussian, SavGol, etc.) that fills a ``TrajectoryRecord`` dict can
use this writer.

HDF5 Schema
-----------
Root-level groups::

    /metadata/          — global processing parameters (attrs)
    /trajectory_{id}/   — one group per trajectory
    /_processing_failures/  — attrs recording error messages for failed tids

Per-trajectory datasets::

    frames              int32   (n_frames,)
    time_vector         float32 (n_frames,)   seconds from first frame
    curvature_matrix    float32 (n_frames, n_resample)   µm⁻¹
    fourier_amplitudes  float32 (n_frames, n_modes)      κ·R or raw
    fourier_phases      float32 (n_frames, n_modes)      radians
    equivalent_radii    float32 (n_frames,)   µm
    perimeters          float32 (n_frames,)   µm
    areas               float32 (n_frames,)   µm²
    centroids           float32 (n_frames, 2)
    original_perimeters float32 (n_frames,)   µm  (from parquet)
    original_areas      float32 (n_frames,)   µm² (from parquet)
    m_max_per_frame     int16   (n_frames,)
    gaussian_sigmas     float32 (n_frames,)   mode-number units
    smoothed_equivalent_radii float32 (n_frames,) µm  (pipeline ≥ 2.0; κ·R radius)
    smoothed_perimeters float32 (n_frames,)   µm  (pipeline ≥ 2.0)
    turning_numbers     float32 (n_frames,)   signed, ≈ ±1 (pipeline ≥ 2.0)
    moment_angle_rad     float32 (n_frames,)   shape orientation, image
                                                2nd-moment tensor, axial
                                                (mod pi), wrapped to
                                                [-pi/2, pi/2)
    ellipse_angle_rad    float32 (n_frames,)   shape orientation,
                                                cv2.fitEllipse, same
                                                convention as above; NaN if
                                                contour had < 5 points
    ellipse_aspect_ratio float32 (n_frames,)   fitted-ellipse major/minor
                                                axis ratio (>= 1); NaN
                                                alongside ellipse_angle_rad

    moment_angle_rad and ellipse_angle_rad are computed directly from the
    raw (x, y) contour points, so — unlike fourier_phases — they are NOT
    affected by the arc-length parameterization's start point.  Use these,
    not the mode-2 Fourier phase, for any comparison to a lab-frame axis
    (e.g. an assumed extensional-flow direction).

Per-trajectory attrs::

    n_frames            int
    n_modes             int
    n_resample          int
    trajectory_id       int

Mandatory global metadata attrs
--------------------------------
The following keys are always written; downstream code may rely on them::

    fft_normalization       str  e.g. "forward"
    amplitude_normalization str  e.g. "kappa_R_um" | "kappa_raw"
    smoothing_method        str  e.g. "spectral_gaussian"
    pixel_size_um           float
    frame_rate_hz           float
    n_resample              int
    lambda_cut_um           float
    processing_timestamp    str  ISO-8601

``pipeline_version`` (absent in files written before 2.0) is also recorded.
Writing into a file that already holds trajectories from a different
``pipeline_version`` raises, so versions are never mixed in one file.

Public API
----------
    open_output(path, params_dict, mode)  -> h5py.File (context manager)
    write_trajectory(h5file, tid, record) -> None
    record_failure(h5file, tid, message)  -> None
    load_trajectory(hdf5_path, tid)       -> dict
    list_trajectories(hdf5_path)          -> list[int]
    get_metadata(hdf5_path)               -> dict
    already_processed_ids(hdf5_path)      -> set[int]
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# Compression used for all datasets
_COMPRESSION = "gzip"
_COMPRESSION_OPTS = 4

# Datasets that must be present in a valid trajectory group
_REQUIRED_DATASETS = (
    "frames",
    "time_vector",
    "curvature_matrix",
    "fourier_amplitudes",
    "fourier_phases",
    "equivalent_radii",
)

# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------


@contextmanager
def open_output(
    path,
    params_dict: dict,
    mode: str = "a",
):
    """Open (or create) an HDF5 output file and write global metadata.

    Usage::

        with open_output(path, params.to_dict()) as h5:
            write_trajectory(h5, tid, record)

    Parameters
    ----------
    path       : file path (str or Path).
    params_dict: dict from ``SpectralPipelineParams.to_dict()`` or equivalent.
    mode       : ``"a"`` (append/create) or ``"w"`` (overwrite).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    h5 = h5py.File(str(path), mode)
    try:
        _ensure_metadata(h5, params_dict)
        yield h5
    finally:
        h5.close()


def _ensure_metadata(h5: h5py.File, params_dict: dict) -> None:
    """Write global metadata if not already present."""
    if "metadata" not in h5:
        grp = h5.create_group("metadata")
    else:
        grp = h5["metadata"]
        existing = grp.attrs.get("pipeline_version", None)
        new = params_dict.get("pipeline_version", None)
        has_traj = any(k.startswith("trajectory_") for k in h5.keys())
        if has_traj and existing != new:
            raise ValueError(
                f"{h5.filename} holds trajectories from pipeline_version "
                f"{existing!r}; refusing to add pipeline_version {new!r}. "
                f"Write to a new output file."
            )

    # Mandatory fields
    mandatory = {
        "fft_normalization": params_dict.get("fft_normalization", "forward"),
        "amplitude_normalization": params_dict.get(
            "amplitude_normalization", "kappa_R_um"
        ),
        "smoothing_method": params_dict.get("smoothing_method", "spectral_gaussian"),
        "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for k, v in {**params_dict, **mandatory}.items():
        grp.attrs[k] = v


# ---------------------------------------------------------------------------
# Trajectory write / read
# ---------------------------------------------------------------------------


def write_trajectory(
    h5file: h5py.File,
    trajectory_id: int,
    record: dict,
    frame_rate: float,
    overwrite: bool = True,
) -> None:
    """Write one trajectory's results to an open HDF5 file.

    Parameters
    ----------
    h5file       : open h5py.File (writable).
    trajectory_id: integer ID.
    record       : dict with keys matching the schema (see module docstring).
                   Arrays need not be pre-typed; they are cast on write.
    frame_rate   : Hz, used to compute ``time_vector`` from ``frames``.
    overwrite    : if True, delete any existing group for this trajectory.
    """
    group_name = f"trajectory_{trajectory_id}"

    if group_name in h5file:
        if overwrite:
            del h5file[group_name]
        else:
            logger.debug(f"Skipping existing {group_name}")
            return

    grp = h5file.create_group(group_name)

    frames = np.asarray(record["frames"], dtype=np.int32)
    n_frames = len(frames)

    time_vector = (frames - frames[0]).astype(np.float32) / frame_rate

    def _save(name: str, data, dtype=np.float32):
        arr = np.asarray(data, dtype=dtype)
        grp.create_dataset(
            name, data=arr, compression=_COMPRESSION, compression_opts=_COMPRESSION_OPTS
        )

    _save("frames", record["frames"], dtype=np.int32)
    _save("time_vector", time_vector, dtype=np.float32)
    _save("curvature_matrix", record["curvature_matrix"], dtype=np.float32)
    _save("fourier_amplitudes", record["fourier_amplitudes"], dtype=np.float32)
    _save("fourier_phases", record["fourier_phases"], dtype=np.float32)
    _save("equivalent_radii", record["equivalent_radii"], dtype=np.float32)
    _save("perimeters", record.get("perimeters", np.zeros(n_frames)), dtype=np.float32)
    _save("areas", record.get("areas", np.zeros(n_frames)), dtype=np.float32)
    _save(
        "m_max_per_frame",
        record.get("m_max_per_frame", np.zeros(n_frames, np.int16)),
        dtype=np.int16,
    )
    _save(
        "gaussian_sigmas",
        record.get("gaussian_sigmas", np.zeros(n_frames)),
        dtype=np.float32,
    )

    # Optional raw-contour geometry (from parquet)
    if "centroids" in record:
        _save("centroids", record["centroids"], dtype=np.float32)
    if "original_perimeters" in record:
        _save("original_perimeters", record["original_perimeters"], dtype=np.float32)
    if "original_areas" in record:
        _save("original_areas", record["original_areas"], dtype=np.float32)

    # Smoothed-curve geometry and validity (pipeline ≥ 2.0)
    for name in ("smoothed_equivalent_radii", "smoothed_perimeters", "turning_numbers"):
        if name in record:
            _save(name, record[name], dtype=np.float32)

    # Optional shape-orientation angles (from raw contour, arc-length-origin-
    # independent — see module docstring)
    if "moment_angle_rad" in record:
        _save("moment_angle_rad", record["moment_angle_rad"], dtype=np.float32)
    if "ellipse_angle_rad" in record:
        _save("ellipse_angle_rad", record["ellipse_angle_rad"], dtype=np.float32)
    if "ellipse_aspect_ratio" in record:
        _save("ellipse_aspect_ratio", record["ellipse_aspect_ratio"], dtype=np.float32)

    # Per-trajectory attrs
    grp.attrs["n_frames"] = n_frames
    grp.attrs["n_modes"] = np.asarray(record["fourier_amplitudes"]).shape[1]
    grp.attrs["n_resample"] = np.asarray(record["curvature_matrix"]).shape[1]
    grp.attrs["trajectory_id"] = int(trajectory_id)


def record_failure(
    h5file: h5py.File,
    trajectory_id: int,
    message: str,
) -> None:
    """Append a failure record to ``/_processing_failures/``."""
    grp = h5file.require_group("_processing_failures")
    # Truncate to avoid huge attrs
    grp.attrs[f"trajectory_{trajectory_id}"] = message[:2000]


# ---------------------------------------------------------------------------
# Read / query helpers
# ---------------------------------------------------------------------------


def load_trajectory(hdf5_path, trajectory_id: int) -> dict:
    """Load all datasets for one trajectory into a dict of numpy arrays.

    Returns
    -------
    dict with keys matching the schema; additionally ``"metadata"`` contains
    the per-trajectory h5py attrs.

    Raises
    ------
    KeyError if the trajectory is not present.
    """
    with h5py.File(str(hdf5_path), "r") as f:
        group_name = f"trajectory_{trajectory_id}"
        if group_name not in f:
            available = list_trajectories(hdf5_path)
            raise KeyError(
                f"Trajectory {trajectory_id} not found in {hdf5_path}. "
                f"Available: {available[:5]}{'...' if len(available) > 5 else ''}"
            )
        grp = f[group_name]
        result = {name: grp[name][:] for name in grp.keys()}
        result["metadata"] = dict(grp.attrs)
        result["trajectory_id"] = int(trajectory_id)

    return result


def list_trajectories(hdf5_path) -> list[int]:
    """Return sorted list of trajectory IDs present in the file."""
    with h5py.File(str(hdf5_path), "r") as f:
        ids = [
            int(k.split("_")[1])
            for k in f.keys()
            if k.startswith("trajectory_") and k != "trajectory_id"
        ]
    return sorted(ids)


def already_processed_ids(hdf5_path) -> set[int]:
    """Return the set of trajectory IDs already written to *hdf5_path*.

    Returns an empty set if the file does not exist.
    """
    path = Path(hdf5_path)
    if not path.exists():
        return set()
    return set(list_trajectories(path))


def get_metadata(hdf5_path) -> dict:
    """Return the global metadata attrs from ``/metadata/``."""
    with h5py.File(str(hdf5_path), "r") as f:
        if "metadata" not in f:
            return {}
        return dict(f["metadata"].attrs)


def get_failure_log(hdf5_path) -> dict[int, str]:
    """Return {trajectory_id: error_message} for all recorded failures."""
    with h5py.File(str(hdf5_path), "r") as f:
        if "_processing_failures" not in f:
            return {}
        grp = f["_processing_failures"]
        return {
            int(k.split("_")[1]): str(v)
            for k, v in grp.attrs.items()
            if k.startswith("trajectory_")
        }


def validate_trajectory_group(hdf5_path, trajectory_id: int) -> list[str]:
    """Check that all required datasets are present.

    Returns a (possibly empty) list of missing dataset names.
    """
    with h5py.File(str(hdf5_path), "r") as f:
        group_name = f"trajectory_{trajectory_id}"
        if group_name not in f:
            return list(_REQUIRED_DATASETS)
        grp = f[group_name]
        return [ds for ds in _REQUIRED_DATASETS if ds not in grp]
