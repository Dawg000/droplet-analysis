"""
contour_io.py
=============
Contour loading from two sources:

  1. On-the-fly RLE decode from a parquet row (primary path).
  2. Pre-built .npz contour cache (optional, accelerates repeated runs).

Cache format (.npz, allow_pickle=False)
---------------------------------------
Compatible with the format expected by pipeline_contour_diagnostics.py::

    frames              : int32 array  shape (n_frames,)  — absolute frame numbers
    contour_{frame_num} : float32 array shape (N, 2)      — one key per frame

One .npz file per trajectory, stored as::

    <cache_dir>/trajectory_{trajectory_id}.npz

Public API
----------
    get_contour_from_rle(rle_string, frame_num, object_id,
                         image_height, image_width)  -> np.ndarray (N,2)

    contour_orientation(contour)                     -> dict
        Shape-orientation angle(s) from raw (x, y) contour points.  See
        the function docstring for why this — not the spectral pipeline's
        mode-2 phase — is the correct field to compare against a lab-frame
        axis such as an assumed extensional-flow direction.

    load_contour_from_cache(cache_dir, trajectory_id, frame_num) -> np.ndarray (N,2)

    write_trajectory_cache(cache_dir, trajectory_id,
                           frame_nums, contours)     -> Path

    cache_exists(cache_dir, trajectory_id)           -> bool

    find_cache_dir(search_root)                      -> Path | None
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RLE decode (CPU-bound; cannot be GPU-accelerated via pycocotools)
# ---------------------------------------------------------------------------


def get_contour_from_rle(
    rle_string,
    frame_num: int,
    object_id: int,
    image_height: int = 800,
    image_width: int = 1280,
) -> np.ndarray:
    """Decode a COCO RLE string to a raw pixel contour.

    Returns the largest connected contour as a (N, 2) float64 array of (x, y)
    pixel coordinates.  Raises ``ValueError`` on any failure so callers can
    skip the frame gracefully.

    Parameters
    ----------
    rle_string:
        COCO RLE counts field — either ``str`` or ``bytes``.
    frame_num:
        Frame number (used only in error messages).
    object_id:
        Object ID (used only in error messages).
    image_height, image_width:
        Sensor dimensions in pixels.
    """
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError("pycocotools is required for RLE decoding") from exc

    try:
        if isinstance(rle_string, str):
            counts = rle_string.encode("utf-8")
        elif isinstance(rle_string, bytes):
            counts = rle_string
        else:
            raise TypeError(
                f"rle_string must be str or bytes, got {type(rle_string).__name__}"
            )

        rle_dict = {"counts": counts, "size": [image_height, image_width]}
        binary_mask = mask_utils.decode(rle_dict)

        if binary_mask is None or not isinstance(binary_mask, np.ndarray):
            raise ValueError("mask_utils.decode returned invalid result")

        if binary_mask.shape != (image_height, image_width):
            raise ValueError(
                f"Mask shape {binary_mask.shape} != expected "
                f"({image_height}, {image_width})"
            )

        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            raise ValueError("No contours found in decoded mask")

        contour = max(contours, key=cv2.contourArea).reshape(-1, 2)

        if len(contour) < 3:
            raise ValueError(
                f"Largest contour has only {len(contour)} points (need ≥ 3)"
            )

        return contour.astype(np.float64)

    except Exception as exc:
        raise ValueError(
            f"RLE decode failed for frame {frame_num}, object {object_id}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Shape orientation
# ---------------------------------------------------------------------------


def contour_orientation(contour: np.ndarray) -> dict:
    """
    Droplet shape-orientation angle from the raw (pre-resample) contour,
    by two independent methods.

    Both angles describe the orientation of the shape's long axis relative
    to the image x-axis, wrapped to [-pi/2, pi/2).  This is *axial* data
    (mod pi, not mod 2*pi): a droplet's elongation direction, like an
    ellipse's major axis, has no distinguishable head or tail.

    Unlike the Fourier mode-2 phase in the spectral pipeline output, these
    angles are computed directly from image-frame (x, y) contour points, so
    they are NOT affected by the arc-length parameterization's start point
    (which is fixed by cv2.findContours' raster-scan order and is not a
    physically meaningful reference).  Use these fields — not the raw mode-2
    phase — for any comparison to a lab-frame axis (e.g. an assumed
    extensional-flow direction).

    Parameters
    ----------
    contour : (N, 2) array, raw pixel (x, y) coordinates.

    Returns
    -------
    dict with keys:
        moment_angle_rad : float
            Orientation of the 2nd-order area-moment (moment-of-inertia)
            tensor.  Well-defined for any closed contour and robust to
            point count and to non-elliptical, irregular boundaries (the
            expected case for droplets distorted by neighbor interactions
            at high volume fraction).  Primary orientation estimate.
        ellipse_angle_rad : float
            Orientation of the least-squares best-fit ellipse
            (cv2.fitEllipse).  Requires >= 5 contour points; NaN otherwise.
            Cross-check against moment_angle_rad.
        ellipse_aspect_ratio : float
            Major/minor axis ratio (>= 1) of the fitted ellipse.  NaN if
            ellipse_angle_rad is NaN.
    All three are NaN if the contour is degenerate (zero area).
    """
    contour = np.asarray(contour, dtype=np.float64)

    result = {
        "moment_angle_rad": np.nan,
        "ellipse_angle_rad": np.nan,
        "ellipse_aspect_ratio": np.nan,
    }

    # --- Image-moment orientation (principal axis of the 2nd-order tensor) ---
    M = cv2.moments(contour.astype(np.float32))
    mu20, mu02, mu11 = M["mu20"], M["mu02"], M["mu11"]
    if M["m00"] > 1e-9 and (abs(mu20 - mu02) > 1e-9 or abs(mu11) > 1e-9):
        angle = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
        result["moment_angle_rad"] = _wrap_axial(angle)

    # --- Ellipse-fit orientation + aspect ratio ---
    if len(contour) >= 5:
        try:
            (_cx, _cy), (d1, d2), angle_deg = cv2.fitEllipse(contour.astype(np.float32))
            # cv2.fitEllipse does not guarantee d1 >= d2 (major, minor); the
            # returned angle always refers to the axis paired with d1, so if
            # d1 is actually the minor axis, rotate by 90 deg to keep the
            # angle pointing along the true major axis.
            if d1 < d2:
                angle_deg += 90.0
                d1, d2 = d2, d1
            if d2 > 1e-9:
                result["ellipse_angle_rad"] = _wrap_axial(np.radians(angle_deg))
                result["ellipse_aspect_ratio"] = float(d1 / d2)
        except cv2.error:
            pass

    return result


def _wrap_axial(angle_rad: float) -> float:
    """Wrap an axial (headless, period-pi) angle to [-pi/2, pi/2)."""
    return float(((angle_rad + np.pi / 2) % np.pi) - np.pi / 2)


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def cache_exists(cache_dir, trajectory_id: int) -> bool:
    """Return True if a cache file exists for *trajectory_id*."""
    return _cache_path(cache_dir, trajectory_id).exists()


def find_cache_dir(search_root) -> Path | None:
    """Walk *search_root* upward then downward looking for a ``contour_cache``
    directory that contains at least one ``.npz`` file.

    Search order:
    1. ``search_root`` itself.
    2. ``search_root / contour_cache``.
    3. ``search_root.parent / contour_cache``.
    4. Any subdirectory named ``contour_cache`` reachable within two levels.

    Returns the first match, or ``None`` if nothing is found.
    """
    root = Path(search_root)
    candidates = [
        root,
        root / "contour_cache",
        root.parent / "contour_cache",
    ]
    # Also probe one level of subdirectories
    try:
        candidates += [p for p in root.iterdir() if p.is_dir()]
        candidates += [p for p in root.parent.iterdir() if p.is_dir()]
    except PermissionError:
        pass

    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("trajectory_*.npz")):
            logger.info(f"Found contour cache at: {candidate}")
            return candidate

    return None


def load_contour_from_cache(
    cache_dir,
    trajectory_id: int,
    frame_num: int,
) -> np.ndarray:
    """Load a single raw contour from the .npz cache.

    Parameters
    ----------
    cache_dir:
        Directory containing ``trajectory_{trajectory_id}.npz`` files.
    trajectory_id:
        Integer trajectory ID.
    frame_num:
        Absolute frame number (not a positional index).

    Returns
    -------
    np.ndarray of shape (N, 2), float64, (x, y) pixel coordinates.
    """
    path = _cache_path(cache_dir, trajectory_id)
    if not path.exists():
        raise FileNotFoundError(f"No cache file for trajectory {trajectory_id}: {path}")

    data = np.load(str(path), allow_pickle=False)
    key = f"contour_{int(frame_num)}"
    if key not in data:
        available = [k for k in data.files if k.startswith("contour_")]
        raise KeyError(
            f"Frame {frame_num} not in cache for trajectory {trajectory_id}. "
            f"Available frames: {len(available)}"
        )
    return data[key].astype(np.float64)


def load_all_contours_from_cache(
    cache_dir,
    trajectory_id: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Load all frames and contours for *trajectory_id* from cache.

    Returns
    -------
    frame_nums : np.ndarray, int32, shape (n_frames,)
    contours   : list of np.ndarray, each shape (N_i, 2), float64
    """
    path = _cache_path(cache_dir, trajectory_id)
    if not path.exists():
        raise FileNotFoundError(f"No cache file for trajectory {trajectory_id}: {path}")

    data = np.load(str(path), allow_pickle=False)
    frame_nums = data["frames"].astype(np.int32)
    contours = [data[f"contour_{int(f)}"].astype(np.float64) for f in frame_nums]
    return frame_nums, contours


def write_trajectory_cache(
    cache_dir,
    trajectory_id: int,
    frame_nums: Sequence[int],
    contours: Sequence[np.ndarray],
) -> Path:
    """Write a contour cache file for one trajectory.

    The file is written atomically (temp file → rename) to avoid partial
    writes if the process is interrupted.

    Parameters
    ----------
    cache_dir:
        Directory to write into (created if absent).
    trajectory_id:
        Integer trajectory ID.
    frame_nums:
        Absolute frame numbers, length n_frames.
    contours:
        Raw pixel contours, one per frame, each shape (N_i, 2).

    Returns
    -------
    Path to the written .npz file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # path = _cache_path(cache_dir, trajectory_id)
    # tmp_path = path.with_suffix(".npz.tmp")

    path = _cache_path(cache_dir, trajectory_id)
    # tmp filename must itself end in ".npz" -- otherwise np.savez_compressed
    # silently appends ".npz" (it doesn't recognize ".npz.tmp" as already
    # having the extension), producing "*.npz.tmp.npz" and breaking the
    # rename below.
    tmp_path = path.with_name(path.name + ".tmp.npz")

    arrays: dict[str, np.ndarray] = {
        "frames": np.asarray(frame_nums, dtype=np.int32),
    }
    for frame_num, contour in zip(frame_nums, contours):
        arrays[f"contour_{int(frame_num)}"] = contour.astype(np.float32)

    np.savez_compressed(str(tmp_path), **arrays)
    tmp_path.rename(path)  # atomic on POSIX

    logger.debug(
        f"Wrote cache for trajectory {trajectory_id}: {len(frame_nums)} frames → {path}"
    )
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cache_path(cache_dir, trajectory_id: int) -> Path:
    return Path(cache_dir) / f"trajectory_{int(trajectory_id)}.npz"
