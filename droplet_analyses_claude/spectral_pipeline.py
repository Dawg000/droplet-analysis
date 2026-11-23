"""
spectral_pipeline.py
====================
Spectral Gaussian pipeline: arc-length resampling → Gaussian-windowed spectral
curvature → Fourier decomposition of κ(s).

All physical quantities are in **microns** (µm).

FFT normalization
-----------------
All transforms use the **"forward"** convention throughout::

    F[k] = (1/N) Σ_n  f[n] exp(-2πi k n / N)

so that ``F[0]`` is the mean of the input and the amplitude spectrum is
independent of N.  This convention is recorded in the HDF5 metadata as
``fft_normalization = "forward"``.

Arc-length parameterization (pipeline_version 2.0)
--------------------------------------------------
Fourier modes of κ are defined on the arc length s of the *smoothed* curve::

    F_m = (1/L) ∮ κ(s) exp(-2πi m s / L) ds

not on the raw-arc-length sample index.  Smoothing cuts corners, so the
windowed curve is traversed non-uniformly; an index-grid FFT over-weights
high-curvature regions (mode 0 exceeded the isoperimetric bound in ~46% of
frames; modes 2 and 4 biased by up to ~10% for faceted shapes).  The integral
is evaluated exactly on a zero-padded grid (see ``_arclength_spectral_batch``).
Hence ``F[0]·R_s = 2πR_s/L_s`` is the smoothed curve's isoperimetric ratio.

Sign convention: κ is multiplied per frame by the sign of the turning number,
so convex regions have κ > 0 regardless of contour traversal direction.
Pipeline < 2.0 stored convex-negative κ (cv2 traversal); every stored phase
differs by π and Ψ(2,4) = 2φ₂ − φ₄ by π between the two versions.

κ·R normalization
-----------------
Fourier amplitudes are stored as dimensionless **κ·R_s** where R_s is the
equivalent radius in **µm** of the smoothed curve (= sqrt(area/π); raw-contour
R is still stored as ``equivalent_radii`` and sets σ and m_max).  This removes the
trivial 1/R dependence of raw curvature amplitudes and allows meaningful
comparisons across droplet sizes.  Recorded in HDF5 metadata as
``amplitude_normalization = "kappa_R_um"``.

m_max
-----
The per-droplet mode cutoff ``m_max = min(floor(R_px), 32)`` (where
``R_px = R_µm / pixel_size``) is stored per-frame in the HDF5 dataset
``m_max_per_frame``.  It is NOT used to truncate the stored amplitudes; all
modes up to N//2 are kept.  ``m_max`` marks the highest mode that can be
considered physically resolved given the contour pixel radius, and is used
downstream when applying the spectral Gaussian filter.

Spectral Gaussian window
------------------------
The window shape in mode space is::

    W[m] = gauss(m) + (1 - gauss(m)) · plateau(m)

where

    gauss(m) = exp(-m² / (2 σ²))
    plateau(m) = 1  if |m| ≤ M_FLAT  else 0

and σ is derived from the physical cutoff wavelength λ_cut (µm) via::

    m_cut = 2π R_µm / λ_cut
    σ     = m_cut / sqrt(ln 2)         (−3 dB point at m_cut)

clamped so that the Nyquist mode retains at most ``nyquist_suppression``
of its unfiltered amplitude.

GPU acceleration
----------------
CuPy is used when available for the batched windowing, curvature,
arc-length reparameterization and Fourier decomposition inside
``process_trajectory_frames``.  The arc-length resampling remains on CPU.  GPU is used only when
the trajectory has enough frames to amortise the transfer overhead; the
threshold is controlled by ``GPU_FRAME_THRESHOLD``.  All functions degrade
gracefully to NumPy/SciPy when CuPy is absent.

Public API
----------
    SpectralPipelineParams          — dataclass of all tunable parameters
    compute_m_max(R_um, pixel_size) -> int
    arc_length_resample(contour, n_out) -> (resampled, L_um, pixel_size)
    gaussian_window(N, sigma, m_flat) -> np.ndarray
    sigma_from_lambda(R_um, lambda_cut_um, N, nyquist_suppression) -> float
    spectral_curvature_and_modes(resampled, R_um, params)
        -> SpectralResult (dataclass)
    process_trajectory_frames(contours, frame_nums, pixel_size, params,
                              use_gpu)
        -> TrajectoryResult (dataclass)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Pipeline version / method identifiers, written to HDF5 global metadata.
# 1.x (unversioned files): κ Fourier modes computed on the raw-arc-length index
#     grid (biased for faceted shapes), negative curvature for convex contours.
# 2.0: modes on the smoothed curve's own arc length, convex = positive κ,
#     κ·R normalized by the smoothed-curve equivalent radius.
PIPELINE_VERSION = "2.0"
CURVATURE_METHOD = "arclength_weighted_spectral"

# GPU threshold: use GPU only when a trajectory has at least this many frames.
GPU_FRAME_THRESHOLD = 20

# Flat passband (modes 0..M_FLAT are held at unity regardless of Gaussian)
M_FLAT = 4

# Nyquist suppression target for sigma clamping
NYQUIST_SUPPRESSION_DEFAULT = 0.01

# Hard cap on stored mode count (per-droplet m_max formula caps at this)
M_MAX_CAP = 32

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
    logger.info("CuPy available — GPU acceleration enabled in spectral_pipeline")
except ImportError:
    _CUPY_AVAILABLE = False
    logger.info("CuPy not available — using CPU (NumPy/SciPy) throughout")


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------


@dataclass
class SpectralPipelineParams:
    """All tunable parameters for the spectral Gaussian pipeline.

    Attributes
    ----------
    pixel_size : float
        Sensor pixel pitch in µm/pixel.
    frame_rate : float
        Acquisition frame rate in Hz.
    n_resample : int
        Number of arc-length-uniform points to resample each contour to.
        Must be even (for clean N//2 mode count).
    lambda_cut_um : float
        Physical −3 dB cutoff wavelength of the Gaussian window in µm.
    m_flat : int
        Modes 0..m_flat are kept at unit gain regardless of the Gaussian
        (flat passband).
    nyquist_suppression : float
        Target Nyquist-mode amplitude after windowing; used to clamp σ.
    image_height, image_width : int
        Sensor dimensions in pixels (passed to RLE decoder).
    normalize_amplitude : bool
        If True, store κ·R amplitudes (dimensionless).  Should always be
        True for publication-quality analysis.
    """

    pixel_size: float = 2.46  # µm / pixel
    frame_rate: float = 800.0  # Hz
    n_resample: int = 64
    lambda_cut_um: float = 16.0  # µm  (−3 dB cutoff wavelength)
    m_flat: int = M_FLAT
    nyquist_suppression: float = NYQUIST_SUPPRESSION_DEFAULT
    image_height: int = 800
    image_width: int = 1280
    normalize_amplitude: bool = True  # store κ·R not raw κ amplitudes

    def __post_init__(self):
        if self.n_resample % 2 != 0:
            raise ValueError(f"n_resample must be even, got {self.n_resample}")
        if self.pixel_size <= 0:
            raise ValueError("pixel_size must be positive")
        if self.lambda_cut_um <= 0:
            raise ValueError("lambda_cut_um must be positive")

    def to_dict(self) -> dict:
        return {
            "pixel_size_um": self.pixel_size,
            "frame_rate_hz": self.frame_rate,
            "n_resample": self.n_resample,
            "lambda_cut_um": self.lambda_cut_um,
            "m_flat": self.m_flat,
            "nyquist_suppression": self.nyquist_suppression,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "normalize_amplitude": int(self.normalize_amplitude),
            # Normalization conventions — consumed by HDF5 writer
            "fft_normalization": "forward",
            "amplitude_normalization": "kappa_R_um"
            if self.normalize_amplitude
            else "kappa_raw",
            "smoothing_method": "spectral_gaussian",
            "pipeline_version": PIPELINE_VERSION,
            "curvature_method": CURVATURE_METHOD,
            "fourier_parameterization": "smoothed_curve_arc_length",
            "curvature_sign_convention": (
                "convex_positive: kappa multiplied per frame by the sign of the "
                "smoothed curve's turning number (pipeline <2.0 stored "
                "convex_negative; stored phases differ by pi, Psi(2,4) by pi)"
            ),
            "normalization_radius": "smoothed_curve_equivalent_radius",
            "curvature_matrix_sampling": "uniform_smoothed_arc_length",
            "upsample_factor": UPSAMPLE,
        }


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SpectralResult:
    """Per-frame result from ``spectral_curvature_and_modes``."""

    curvature: np.ndarray  # shape (n_resample,), µm⁻¹, uniform smoothed arc length
    fourier_amplitudes: np.ndarray  # shape (n_modes,)  κ·R_s or raw κ
    fourier_phases: np.ndarray  # shape (n_modes,)  radians
    equivalent_radius_um: float  # raw-contour √(area/π) in µm
    perimeter_um: float  # smoothed-curve arc length, µm
    m_max: int  # per-droplet mode cutoff
    gaussian_sigma: float  # σ actually used
    smoothed_equivalent_radius_um: float  # κ·R normalization radius, µm
    turning_number: float  # signed, ≈ ±1 for a simple smoothed curve


@dataclass
class TrajectoryResult:
    """Per-trajectory result from ``process_trajectory_frames``."""

    frames: np.ndarray  # int32, shape (n_frames,)
    curvature_matrix: np.ndarray  # float32, (n_frames, n_resample), uniform smoothed s
    fourier_amplitudes: np.ndarray  # float32, (n_frames, n_modes), κ·R_s
    fourier_phases: np.ndarray  # float32, shape (n_frames, n_modes)
    equivalent_radii_um: np.ndarray  # float32, (n_frames,)  raw contour
    perimeters_um: np.ndarray  # float32, (n_frames,)  raw contour
    m_max_per_frame: np.ndarray  # int16,   shape (n_frames,)
    gaussian_sigmas: np.ndarray  # float32, shape (n_frames,)
    # Geometry from the raw (pre-resample) contour
    areas_um2: np.ndarray  # float32, shape (n_frames,)
    # Geometry of the smoothed (windowed) curve the spectra describe
    smoothed_equivalent_radii_um: np.ndarray  # float32, (n_frames,)
    smoothed_perimeters_um: np.ndarray  # float32, (n_frames,)
    turning_numbers: np.ndarray  # float32, (n_frames,) signed, ≈ ±1


# ---------------------------------------------------------------------------
# m_max
# ---------------------------------------------------------------------------


def compute_m_max(R_um: float, pixel_size: float) -> int:
    """Per-droplet spectral mode cutoff.

    ``m_max = min(floor(R_px), M_MAX_CAP)``

    where ``R_px = R_um / pixel_size``.  This is the highest mode that can be
    considered physically resolved given the contour pixel radius; it is stored
    in the HDF5 output but does **not** truncate the stored amplitude array.
    """
    R_px = R_um / pixel_size
    return min(int(np.floor(R_px)), M_MAX_CAP)


# ---------------------------------------------------------------------------
# Arc-length resampling
# ---------------------------------------------------------------------------


def arc_length_resample(
    contour: np.ndarray,
    n_out: int,
    pixel_size: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Resample *contour* to *n_out* uniformly arc-length-spaced points.

    Parameters
    ----------
    contour : (N, 2) float, pixel coordinates (x, y).
    n_out   : target number of output points.
    pixel_size : µm / pixel; used to convert perimeter to µm.

    Returns
    -------
    resampled : (n_out, 2) float64, pixel coordinates.
    perimeter_um : total perimeter in µm.
    """
    x = contour[:, 0].astype(np.float64)
    y = contour[:, 1].astype(np.float64)

    dx = np.diff(x, append=x[0])
    dy = np.diff(y, append=y[0])
    seg = np.sqrt(dx**2 + dy**2)

    arc = np.concatenate([[0.0], np.cumsum(seg[:-1])])
    L_px = float(np.sum(seg))

    s = np.linspace(0.0, L_px, n_out, endpoint=False)

    # Extend arrays by one for linear interp to handle wrap
    arc_ext = np.append(arc, L_px)
    x_ext = np.append(x, x[0])
    y_ext = np.append(y, y[0])

    x_r = np.interp(s, arc_ext, x_ext)
    y_r = np.interp(s, arc_ext, y_ext)

    return np.column_stack([x_r, y_r]), L_px * pixel_size


# ---------------------------------------------------------------------------
# Gaussian window
# ---------------------------------------------------------------------------


def gaussian_window(N: int, sigma: float, m_flat: int = M_FLAT) -> np.ndarray:
    """Spectral Gaussian window with flat-passband correction.

    Parameters
    ----------
    N      : contour length (number of resampled points).
    sigma  : Gaussian width in mode-number units.
    m_flat : modes 0..|m_flat| are forced to unity gain.

    Returns
    -------
    W : (N,) float64 window array suitable for element-wise multiply with
        the full (not rfft) frequency array.
    """
    m = np.fft.fftfreq(N) * N  # mode indices: 0, 1, …, N//2, -N//2+1, …
    gauss = np.exp(-(m**2) / (2.0 * sigma**2))
    plateau = (np.abs(m) <= m_flat).astype(np.float64)
    return gauss + (1.0 - gauss) * plateau


def sigma_from_lambda(
    R_um: float,
    lambda_cut_um: float,
    N: int,
    nyquist_suppression: float = NYQUIST_SUPPRESSION_DEFAULT,
) -> float:
    """Compute per-droplet Gaussian σ from physical cutoff wavelength.

    The −3 dB mode is ``m_cut = 2π R_µm / λ_cut``, and
    ``σ = m_cut / sqrt(ln 2)`` places the −3 dB point exactly at m_cut.

    σ is clamped so that the Nyquist mode (m = N/2) retains at most
    ``nyquist_suppression`` of its amplitude::

        σ_max = (N/2) / sqrt(2 ln(1 / nyquist_suppression))

    Parameters
    ----------
    R_um            : equivalent radius in µm.
    lambda_cut_um   : −3 dB wavelength in µm.
    N               : number of resampled contour points.
    nyquist_suppression : target Nyquist amplitude (default 0.01 = −40 dB).

    Returns
    -------
    sigma : float, in mode-number units.
    """
    if lambda_cut_um <= 0 or R_um <= 0:
        # Degenerate: pass-all (very wide Gaussian)
        return (N / 4.0) / np.sqrt(np.log(2.0))

    m_cut = 2.0 * np.pi * R_um / lambda_cut_um
    sigma = m_cut / np.sqrt(np.log(2.0))

    sigma_max = (N / 2.0) / np.sqrt(2.0 * np.log(1.0 / nyquist_suppression))
    return float(np.clip(sigma, 1.0, sigma_max))


# ---------------------------------------------------------------------------
# Core: arc-length-weighted spectral curvature + Fourier decomposition
# ---------------------------------------------------------------------------
#
# The resampled contour is uniform in *raw-polygon* arc length.  After the
# Gaussian window is applied, the smoothed curve is no longer traversed at
# uniform speed (smoothing cuts corners, so speed drops exactly where
# curvature is concentrated).  Fourier modes of κ are therefore defined on the
# smoothed curve's OWN arc length s:
#
#     F_m = (1/L) ∮ κ(s) exp(-2πi m s / L) ds
#         = (1/L) ∫ κ(t) v(t) exp(-2πi m s(t) / L) dt,     v = |dz/dt|
#
# The windowed curve is an exact trigonometric polynomial in t, so z, z', z''
# are evaluated exactly on a zero-padded grid (UPSAMPLE × N points) and the
# integral is done by the trapezoid rule, which converges spectrally for a
# smooth periodic integrand.  No interpolation enters the Fourier modes.
#
# Consequences by construction:
#   * F_0 = 2π·(turning number)/L; after sign normalization F_0·R_s = 2πR_s/L_s
#     is the isoperimetric ratio of the smoothed curve, ≤ 1 for a simple curve.
#   * The per-frame turning number (≈ ±1) is returned as a validity check;
#     |turning| far from 1 flags a self-intersecting smoothed curve.

# Zero-padding factor for evaluating the band-limited windowed curve.
UPSAMPLE = 8

# Frames whose |turning number| deviates from 1 by more than this are logged.
TURNING_TOL = 0.05


def _arclength_spectral_batch(
    resampled: np.ndarray,  # (n_frames, N, 2) px, uniform in raw arc length
    sigmas: np.ndarray,  # (n_frames,) Gaussian σ in mode units
    m_flat: int,
    xp,  # numpy or cupy module
) -> dict:
    """Arc-length-weighted spectral curvature for a batch of frames.

    All arithmetic in float64/complex128 on the array module ``xp``.

    Returns (all numpy arrays, pixel units)
    -------
    kappa_uniform_s : (n_frames, N) curvature in px⁻¹, sign-normalized
                      (convex = positive), sampled at N points uniform in the
                      smoothed curve's arc length, starting at the contour's
                      start point.
    F               : (n_frames, N//2 + 1) complex, forward-normalized
                      arc-length Fourier coefficients of κ (px⁻¹).
    L_px            : (n_frames,) smoothed-curve perimeter in px.
    R_px            : (n_frames,) smoothed-curve equivalent radius √(A/π) in px.
    turning         : (n_frames,) signed turning number before normalization.
    """
    n_frames, N, _ = resampled.shape
    h = N // 2
    Nu = UPSAMPLE * N
    dt = 1.0 / UPSAMPLE  # t measured in original sample-index units

    z = xp.asarray(resampled[..., 0] + 1j * resampled[..., 1], dtype=xp.complex128)
    m = xp.asarray(np.fft.fftfreq(N) * N)  # (N,)

    # Window (same definition as gaussian_window), per-frame σ
    sig = xp.asarray(sigmas, dtype=xp.float64)[:, None]
    gauss = xp.exp(-(m[None, :] ** 2) / (2.0 * sig**2))
    plateau = (xp.abs(m) <= m_flat).astype(xp.float64)[None, :]
    W = gauss + (1.0 - gauss) * plateau
    W[:, h] = 0.0  # drop the ambiguous Nyquist bin (already ≤ nyquist_suppression)

    Zw = W * xp.fft.fft(z, axis=1)
    D1 = (2j * np.pi / N) * m[None, :]

    def _eval(Zk):
        """Zero-pad spectrum to Nu bins and return values on the fine t grid."""
        Zp = xp.zeros((n_frames, Nu), dtype=xp.complex128)
        Zp[:, :h] = Zk[:, :h]
        Zp[:, Nu - (h - 1) :] = Zk[:, h + 1 :]
        return xp.fft.ifft(Zp, axis=1) * UPSAMPLE

    zf = _eval(Zw)
    z1 = _eval(D1 * Zw)
    z2 = _eval(D1 * D1 * Zw)

    v = xp.abs(z1)
    v = xp.where(v < 1e-12, 1e-12, v)
    cross = xp.imag(xp.conj(z1) * z2)  # x'y'' − y'x''
    kappa = cross / v**3  # px⁻¹, sign set by traversal

    L = xp.sum(v, axis=1) * dt  # (n_frames,)
    turning = xp.sum(cross / v**2, axis=1) * dt / (2.0 * np.pi)
    sgn = xp.where(turning < 0, -1.0, 1.0)
    kappa = kappa * sgn[:, None]

    area = 0.5 * xp.abs(xp.sum(xp.imag(xp.conj(zf) * z1), axis=1)) * dt
    R = xp.sqrt(area / np.pi)

    # Arc length s(t) = ∫v dt, integrated spectrally (a cumsum is only
    # first-order accurate and limits the whole quadrature to O(dt)):
    # s = v̄·t + ∫(v − v̄) dt, the periodic part via division by iω.
    t = xp.arange(Nu, dtype=xp.float64) * dt
    wk = 2.0 * np.pi * xp.asarray(np.fft.fftfreq(Nu, d=dt))  # rad per index unit
    Vk = xp.fft.fft(v, axis=1)
    inv = xp.zeros(Nu, dtype=xp.complex128)
    nz = wk != 0
    inv[nz] = 1.0 / (1j * wk[nz])
    inv[Nu // 2] = 0.0  # Nyquist
    s_per = xp.real(xp.fft.ifft(Vk * inv[None, :], axis=1))
    s = (L / N)[:, None] * t[None, :] + s_per - s_per[:, :1]

    # Fourier modes: w_j = κ v dt / L, F_m = Σ_j w_j exp(-i m θ_j), θ = 2π s/L.
    # Powers of exp(-iθ) by recurrence (n_modes ≤ 33, unit-modulus, float64).
    n_modes = h + 1
    w = kappa * v * dt / L[:, None]
    e = xp.exp(-2j * np.pi * s / L[:, None])
    F = xp.empty((n_frames, n_modes), dtype=xp.complex128)
    acc = w.astype(xp.complex128)
    F[:, 0] = xp.sum(acc, axis=1)
    for k in range(1, n_modes):
        acc = acc * e
        F[:, k] = xp.sum(acc, axis=1)

    # κ at N points uniform in s: periodic linear interpolation on the fine
    # grid, batched by offsetting each row onto a single monotone axis.
    Lmax = float(xp.max(L)) * 2.0 + 1.0
    offs = xp.arange(n_frames, dtype=xp.float64)[:, None] * Lmax
    s_ext = xp.concatenate([s, L[:, None]], axis=1) + offs
    k_ext = xp.concatenate([kappa, kappa[:, :1]], axis=1)
    s_q = xp.arange(N, dtype=xp.float64)[None, :] * (L[:, None] / N) + offs
    k_uni = xp.interp(s_q.ravel(), s_ext.ravel(), k_ext.ravel()).reshape(n_frames, N)

    to_np = (lambda a: a.get()) if xp is not np else (lambda a: a)
    return {
        "kappa_uniform_s": to_np(k_uni),
        "F": to_np(F),
        "L_px": to_np(L),
        "R_px": to_np(R),
        "turning": to_np(turning),
    }


def _arclength_spectral(resampled, sigmas, m_flat, use_gpu):
    """Dispatch to CuPy when requested/available, falling back to NumPy."""
    if use_gpu and _CUPY_AVAILABLE:
        try:
            return _arclength_spectral_batch(resampled, sigmas, m_flat, cp)
        except Exception as exc:
            logger.warning(
                f"GPU spectral curvature failed ({exc}), falling back to CPU"
            )
    return _arclength_spectral_batch(resampled, sigmas, m_flat, np)


# ---------------------------------------------------------------------------
# Per-frame entry point
# ---------------------------------------------------------------------------


def spectral_curvature_and_modes(
    resampled: np.ndarray,
    R_um: float,
    params: SpectralPipelineParams,
) -> SpectralResult:
    """Full spectral pipeline for one contour frame (CPU).

    Single-frame wrapper around the same batch core used by
    ``process_trajectory_frames``, so both entry points give identical results.

    Parameters
    ----------
    resampled : (n_resample, 2) float64, raw-arc-length-uniform pixel coords.
    R_um      : raw-contour equivalent radius in µm (sets σ and m_max).
    params    : SpectralPipelineParams.
    """
    N = len(resampled)
    sigma = sigma_from_lambda(R_um, params.lambda_cut_um, N, params.nyquist_suppression)
    out = _arclength_spectral(resampled[None], np.array([sigma]), params.m_flat, False)

    R_s_um = float(out["R_px"][0]) * params.pixel_size
    F_um = out["F"][0] / params.pixel_size  # px⁻¹ → µm⁻¹
    amps = np.abs(F_um) * (R_s_um if params.normalize_amplitude else 1.0)

    return SpectralResult(
        curvature=out["kappa_uniform_s"][0] / params.pixel_size,
        fourier_amplitudes=amps,
        fourier_phases=np.angle(F_um),
        equivalent_radius_um=R_um,
        perimeter_um=float(out["L_px"][0]) * params.pixel_size,
        m_max=compute_m_max(R_um, params.pixel_size),
        gaussian_sigma=sigma,
        smoothed_equivalent_radius_um=R_s_um,
        turning_number=float(out["turning"][0]),
    )


# ---------------------------------------------------------------------------
# Trajectory-level processor
# ---------------------------------------------------------------------------


def process_trajectory_frames(
    contours: Sequence[np.ndarray],
    frame_nums: Sequence[int],
    R_um_per_frame: Sequence[float],
    areas_um2_per_frame: Sequence[float],
    params: SpectralPipelineParams,
    use_gpu: bool = True,
) -> TrajectoryResult:
    """Process all frames of one trajectory through the spectral pipeline.

    Arc-length resampling and σ selection run per-frame on CPU; windowing,
    curvature, arc-length reparameterization and the Fourier decomposition run
    batched over all frames, on the GPU when CuPy is available,
    ``use_gpu=True`` and the trajectory has ≥ GPU_FRAME_THRESHOLD frames.

    Parameters
    ----------
    contours          : raw pixel contours, one (N_i, 2) array per frame.
    frame_nums        : absolute frame numbers, length n_frames.
    R_um_per_frame    : raw-contour equivalent radius in µm per frame
                        (sets σ and m_max; stored as ``equivalent_radii_um``).
    areas_um2_per_frame : raw-contour area in µm² per frame.
    params            : SpectralPipelineParams.
    use_gpu           : allow GPU if CuPy is available.

    Returns
    -------
    TrajectoryResult
    """
    n_frames = len(contours)
    if n_frames == 0:
        raise ValueError("No frames provided to process_trajectory_frames")

    N = params.n_resample

    resampled = np.empty((n_frames, N, 2), dtype=np.float64)
    perimeters_um = np.empty(n_frames, dtype=np.float32)
    m_max_arr = np.empty(n_frames, dtype=np.int16)
    sigma_arr = np.empty(n_frames, dtype=np.float64)

    for i, (contour, R_um) in enumerate(zip(contours, R_um_per_frame)):
        resampled[i], peri_um = arc_length_resample(contour, N, params.pixel_size)
        perimeters_um[i] = peri_um
        sigma_arr[i] = sigma_from_lambda(
            R_um, params.lambda_cut_um, N, params.nyquist_suppression
        )
        m_max_arr[i] = compute_m_max(R_um, params.pixel_size)

    batch_gpu = use_gpu and _CUPY_AVAILABLE and n_frames >= GPU_FRAME_THRESHOLD
    out = _arclength_spectral(resampled, sigma_arr, params.m_flat, batch_gpu)

    bad = np.abs(np.abs(out["turning"]) - 1.0) > TURNING_TOL
    if bad.any():
        logger.warning(
            f"{int(bad.sum())}/{n_frames} frames have |turning number| "
            f"outside 1±{TURNING_TOL} (self-intersecting smoothed curve?)"
        )

    R_s_um = out["R_px"] * params.pixel_size
    F_um = out["F"] / params.pixel_size  # px⁻¹ → µm⁻¹
    amps = np.abs(F_um)
    if params.normalize_amplitude:
        amps = amps * R_s_um[:, None]

    return TrajectoryResult(
        frames=np.asarray(frame_nums, dtype=np.int32),
        curvature_matrix=(out["kappa_uniform_s"] / params.pixel_size).astype(
            np.float32
        ),
        fourier_amplitudes=amps.astype(np.float32),
        fourier_phases=np.angle(F_um).astype(np.float32),
        equivalent_radii_um=np.asarray(R_um_per_frame, dtype=np.float32),
        perimeters_um=perimeters_um,
        m_max_per_frame=m_max_arr,
        gaussian_sigmas=sigma_arr.astype(np.float32),
        areas_um2=np.asarray(areas_um2_per_frame, dtype=np.float32),
        smoothed_equivalent_radii_um=R_s_um.astype(np.float32),
        smoothed_perimeters_um=(out["L_px"] * params.pixel_size).astype(np.float32),
        turning_numbers=out["turning"].astype(np.float32),
    )
