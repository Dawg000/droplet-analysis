import gzip
import logging
import multiprocessing as mp
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

# GPU acceleration imports with fallbacks
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


@dataclass
class BlobFeatures:
    """Container for comprehensive blob features"""

    # Basic geometric
    area: float
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    aspect_ratio: float
    solidity: float
    extent: float

    # Shape descriptors
    perimeter: float
    circularity: float
    eccentricity: float
    major_axis: float
    minor_axis: float
    orientation: float
    roughness: float
    compactness: float
    convexity_defect_count: int
    max_defect_depth: float
    moment_orientation: float

    # Hu moments (7 invariant features)
    hu_moment_1: float
    hu_moment_2: float
    hu_moment_3: float
    hu_moment_4: float
    hu_moment_5: float
    hu_moment_6: float
    hu_moment_7: float

    # Central moments
    central_moment_20: float
    central_moment_02: float
    central_moment_11: float

    # Focus/sharpness metrics
    sobel_mean: float
    sobel_std: float
    laplacian_var: float
    edge_strength: float
    brenner_gradient: float
    tenengrad_var: float
    image_entropy: float

    # Intensity statistics
    intensity_mean: float
    intensity_std: float
    intensity_skewness: float
    intensity_kurtosis: float

    # Quality indicators
    is_in_focus: bool
    focus_score: float

    # RLE Counts Str
    rle_counts_str: str


class GPUAccelerator:
    """Handles different GPU acceleration backends"""

    def __init__(self, prefer_backend="torch"):
        self.backend = None
        self.device = None

        if prefer_backend == "torch" and TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    self.backend = "torch"
                    self.device = torch.device("cuda")
                    logging.info(
                        f"Using PyTorch GPU acceleration on {torch.cuda.get_device_name()}"
                    )
                else:
                    logging.info("PyTorch available but no CUDA device found")
            except Exception as e:
                logging.warning(f"PyTorch GPU initialization failed: {e}")

        if self.backend is None and CUPY_AVAILABLE:
            try:
                test_array = cp.array([1, 2, 3])
                _ = cp.sum(test_array)
                self.backend = "cupy"
                logging.info("Using CuPy GPU acceleration")
            except Exception as e:
                logging.warning(f"CuPy initialization failed: {e}")

        if self.backend is None:
            logging.info("Using CPU processing only")

    def to_gpu(self, array):
        """Move array to GPU"""
        if self.backend == "torch":
            return torch.from_numpy(array).to(self.device)
        elif self.backend == "cupy":
            return cp.asarray(array)
        else:
            return array

    def to_cpu(self, array):
        """Move array back to CPU"""
        if self.backend == "torch":
            return array.cpu().numpy()
        elif self.backend == "cupy":
            return cp.asnumpy(array)
        else:
            return array

    def compute_stats(self, masked_data):
        """Compute statistics on GPU"""
        if len(masked_data) == 0:
            return {"mean": 0, "std": 0, "var": 0}

        if self.backend == "torch":
            mean_val = torch.mean(masked_data).item()
            std_val = torch.std(masked_data).item()
            var_val = torch.var(masked_data).item()
        elif self.backend == "cupy":
            mean_val = float(cp.mean(masked_data))
            std_val = float(cp.std(masked_data))
            var_val = float(cp.var(masked_data))
        else:
            mean_val = np.mean(masked_data)
            std_val = np.std(masked_data)
            var_val = np.var(masked_data)

        return {"mean": mean_val, "std": std_val, "var": var_val}


class SAM2BlobAnalyzer:
    """Efficient multi-GPU blob analysis for SAM2 detection results with parallelization"""

    def __init__(
        self,
        focus_threshold: float = 100.0,
        min_area: int = 50,
        max_area: int = 10000,
        redundancy_iou_threshold: float = 0.7,
        use_gpu: bool = True,
        gpu_backend: str = "torch",
        n_workers: int = None,
    ):
        """
        Initialize the analyzer

        Args:
            focus_threshold: Laplacian variance threshold for in-focus detection
            min_area: Minimum blob area to consider
            max_area: Maximum blob area to consider
            redundancy_iou_threshold: IoU threshold for redundant mask detection
            use_gpu: Whether to use GPU acceleration
            gpu_backend: Preferred GPU backend ('torch', 'cupy', or 'auto')
            n_workers: Number of parallel workers (None = auto-detect)
        """
        self.focus_threshold = focus_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.redundancy_threshold = redundancy_iou_threshold

        # Setup GPU acceleration
        if use_gpu:
            self.gpu = GPUAccelerator(prefer_backend=gpu_backend)
            self.use_gpu = self.gpu.backend is not None
        else:
            self.gpu = None
            self.use_gpu = False

        # Setup parallel processing
        self.n_workers = n_workers or min(mp.cpu_count(), 8)
        logging.info(f"Initialized with {self.n_workers} workers")

    def load_detection_file(self, filepath: str) -> Dict:
        """Load SAM2 detection results from pkl.gz file"""
        with gzip.open(filepath, "rb") as f:
            return pickle.load(f)

    def rle_to_mask(self, rle_data: Dict, height: int, width: int) -> np.ndarray:
        """Convert RLE encoding to binary mask"""
        if isinstance(rle_data, dict) and "counts" in rle_data:
            # Handle COCO RLE format
            try:
                from pycocotools import mask as mask_utils

                return mask_utils.decode(rle_data)
            except ImportError:
                logging.warning("pycocotools not available for RLE decoding")
                return np.zeros((height, width), dtype=np.uint8)
        else:
            # Assume it's already a binary mask
            return (
                rle_data.astype(np.uint8)
                if hasattr(rle_data, "dtype")
                else np.array(rle_data, dtype=np.uint8)
            )

    def compute_basic_features(self, mask: np.ndarray) -> Dict:
        """Compute basic geometric features (unified entry point)"""
        if self.use_gpu:
            return self.compute_basic_features_gpu(mask)
        else:
            return self.compute_basic_features_cpu(mask)

    def compute_basic_features_gpu(self, mask: np.ndarray) -> Dict:
        """Compute basic geometric features with GPU acceleration"""
        if not self.use_gpu:
            return self.compute_basic_features_cpu(mask)

        mask_gpu = self.gpu.to_gpu(mask.astype(np.float32))

        if self.gpu.backend == "torch":
            area = torch.sum(mask_gpu).item()
            if area == 0:
                return None

            # Find coordinates
            coords = torch.nonzero(mask_gpu)
            if len(coords) == 0:
                return None

            y_coords, x_coords = coords[:, 0], coords[:, 1]
            centroid_x = torch.mean(x_coords.float()).item()
            centroid_y = torch.mean(y_coords.float()).item()

            x_min, x_max = torch.min(x_coords).item(), torch.max(x_coords).item()
            y_min, y_max = torch.min(y_coords).item(), torch.max(y_coords).item()

        elif self.gpu.backend == "cupy":
            area = float(cp.sum(mask_gpu))
            if area == 0:
                return None

            y_coords, x_coords = cp.where(mask_gpu > 0)
            centroid_x = float(cp.mean(x_coords))
            centroid_y = float(cp.mean(y_coords))

            x_min, x_max = int(cp.min(x_coords)), int(cp.max(x_coords))
            y_min, y_max = int(cp.min(y_coords)), int(cp.max(y_coords))

        bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
        aspect_ratio = bbox[3] / bbox[2] if bbox[2] > 0 else 1.0
        extent = area / (bbox[2] * bbox[3])

        return {
            "area": area,
            "centroid": (centroid_x, centroid_y),
            "bbox": bbox,
            "aspect_ratio": aspect_ratio,
            "extent": extent,
        }

    def compute_focus_features(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """Unified entry point for focus feature computation"""
        if self.use_gpu:
            return self.compute_focus_features_gpu(image, mask)
        else:
            return self.compute_focus_features_cpu(image, mask)

    def compute_basic_features_cpu(self, mask: np.ndarray) -> Dict:
        """CPU fallback for basic features"""
        area = float(np.sum(mask))
        if area == 0:
            return None

        y_coords, x_coords = np.where(mask > 0)
        centroid_x = float(np.mean(x_coords))
        centroid_y = float(np.mean(y_coords))

        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

        bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
        aspect_ratio = bbox[3] / bbox[2] if bbox[2] > 0 else 1.0
        extent = area / (bbox[2] * bbox[3])

        return {
            "area": area,
            "centroid": (centroid_x, centroid_y),
            "bbox": bbox,
            "aspect_ratio": aspect_ratio,
            "extent": extent,
        }

    def compute_shape_features(self, mask: np.ndarray) -> Dict:
        """Compute comprehensive shape-based features using OpenCV with robust error handling"""
        try:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return self._get_default_shape_features()

            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if area == 0 or perimeter == 0:
                return self._get_default_shape_features()

            # Basic shape metrics
            circularity = 4 * np.pi * area / (perimeter**2)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Convexity defects with robust error handling
            convexity_defect_count = 0
            max_defect_depth = 0

            try:
                hull_indices = cv2.convexHull(contour, returnPoints=False)
                if len(hull_indices) > 3 and len(contour) >= 5:
                    # Additional validation for hull indices
                    if self._validate_hull_indices(hull_indices, len(contour)):
                        defects = cv2.convexityDefects(contour, hull_indices)
                        if defects is not None:
                            convexity_defect_count = len(defects)
                            max_defect_depth = np.max(defects[:, 0, 3]) / 256.0
                        else:
                            convexity_defect_count = 0
                            max_defect_depth = 0
                    else:
                        logging.debug(
                            "Invalid hull indices detected, skipping convexity defects for this mask"
                        )
                else:
                    logging.debug(
                        f"Insufficient points for convexity defects: hull={len(hull_indices) if 'hull_indices' in locals() else 'N/A'}, contour={len(contour)}"
                    )

            except cv2.error as e:
                if "monotonous" in str(e) or "self-intersections" in str(e):
                    logging.debug(
                        f"Skipping convexity defects due to self-intersecting contour: {e}"
                    )
                    # Try to clean the contour and retry once
                    cleaned_contour = self._clean_contour(contour)
                    if cleaned_contour is not None and len(cleaned_contour) != len(
                        contour
                    ):
                        try:
                            hull_indices_clean = cv2.convexHull(
                                cleaned_contour, returnPoints=False
                            )
                            if len(hull_indices_clean) > 3:
                                defects = cv2.convexityDefects(
                                    cleaned_contour, hull_indices_clean
                                )
                                if defects is not None:
                                    convexity_defect_count = len(defects)
                                    max_defect_depth = np.max(defects[:, 0, 3]) / 256.0
                                    logging.debug(
                                        "Successfully computed convexity defects after contour cleaning"
                                    )
                        except cv2.error:
                            logging.debug(
                                "Contour cleaning failed, using default values for convexity defects"
                            )
                else:
                    logging.warning(
                        f"Unexpected OpenCV error in convexity defects: {e}"
                    )
            except Exception as e:
                logging.warning(f"Unexpected error computing convexity defects: {e}")

            # Ellipse fitting with error handling
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    major_axis = max(ellipse[1])
                    minor_axis = min(ellipse[1])
                    eccentricity = (
                        np.sqrt(1 - (minor_axis / major_axis) ** 2)
                        if major_axis > 0
                        else 0
                    )
                    orientation = ellipse[2]
                except cv2.error as e:
                    logging.debug(f"Ellipse fitting failed: {e}")
                    major_axis = minor_axis = eccentricity = orientation = 0
            else:
                major_axis = minor_axis = eccentricity = orientation = 0

            # Hu moments with error handling
            try:
                moments = cv2.moments(mask)
                hu_moments = cv2.HuMoments(moments).flatten()
                hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            except Exception as e:
                logging.debug(f"Hu moments computation failed: {e}")
                hu_moments = np.zeros(7)

            # Additional metrics
            hull_perimeter = cv2.arcLength(hull, True)
            roughness = perimeter / hull_perimeter if hull_perimeter > 0 else 1
            compactness = (perimeter**2) / area if area > 0 else float("inf")

            # Central moments with error handling
            try:
                if moments["m00"] > 0:
                    mu20 = moments["mu20"] / moments["m00"]
                    mu02 = moments["mu02"] / moments["m00"]
                    mu11 = moments["mu11"] / moments["m00"]
                    moment_orientation = (
                        0.5 * np.arctan2(2 * mu11, mu20 - mu02) * 180 / np.pi
                    )
                else:
                    mu20 = mu02 = mu11 = moment_orientation = 0
            except:
                mu20 = mu02 = mu11 = moment_orientation = 0

            return {
                "perimeter": perimeter,
                "circularity": circularity,
                "solidity": solidity,
                "eccentricity": eccentricity,
                "major_axis": major_axis,
                "minor_axis": minor_axis,
                "orientation": orientation,
                "roughness": roughness,
                "compactness": compactness,
                "convexity_defect_count": convexity_defect_count,
                "max_defect_depth": max_defect_depth,
                "moment_orientation": moment_orientation,
                "hu_moment_1": hu_moments[0],
                "hu_moment_2": hu_moments[1],
                "hu_moment_3": hu_moments[2],
                "hu_moment_4": hu_moments[3],
                "hu_moment_5": hu_moments[4],
                "hu_moment_6": hu_moments[5],
                "hu_moment_7": hu_moments[6],
                "central_moment_20": mu20,
                "central_moment_02": mu02,
                "central_moment_11": mu11,
            }

        except Exception as e:
            logging.error(f"Critical error in compute_shape_features: {e}")
            return self._get_default_shape_features()

    def _validate_hull_indices(
        self, hull_indices: np.ndarray, contour_length: int
    ) -> bool:
        """
        Validate that hull indices are monotonous and within bounds

        Args:
            hull_indices: Hull indices from cv2.convexHull
            contour_length: Length of the original contour

        Returns:
            True if indices are valid, False otherwise
        """
        try:
            # Check if indices are within bounds
            if np.any(hull_indices >= contour_length) or np.any(hull_indices < 0):
                return False

            # Check if indices are monotonous (required by convexityDefects)
            hull_flat = hull_indices.flatten()
            if len(hull_flat) < 3:
                return False

            # For a valid convex hull, indices should be in a specific order
            # Check for basic monotonicity issues that cause the error
            sorted_indices = np.sort(hull_flat)
            if not np.array_equal(hull_flat, sorted_indices):
                # Check if it's just reversed order
                if np.array_equal(hull_flat, sorted_indices[::-1]):
                    return True
                # Check for more complex ordering issues
                unique_indices = np.unique(hull_flat)
                if len(unique_indices) < len(hull_flat):
                    return False  # Duplicate indices

            return True

        except Exception:
            return False

    def _clean_contour(self, contour: np.ndarray) -> np.ndarray:
        """
        Attempt to clean a contour to remove self-intersections

        Args:
            contour: Input contour

        Returns:
            Cleaned contour or None if cleaning failed
        """
        try:
            # Simple approach: use approxPolyDP to smooth the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            cleaned = cv2.approxPolyDP(contour, epsilon, True)

            # Ensure minimum number of points
            if len(cleaned) < 5:
                return None

            return cleaned

        except Exception:
            return None

    def _get_default_shape_features(self) -> Dict:
        """Return default shape features when computation fails"""
        return {
            "perimeter": 0,
            "circularity": 0,
            "solidity": 0,
            "eccentricity": 0,
            "major_axis": 0,
            "minor_axis": 0,
            "orientation": 0,
            "roughness": 1,
            "compactness": float("inf"),
            "convexity_defect_count": 0,
            "max_defect_depth": 0,
            "moment_orientation": 0,
            "hu_moment_1": 0,
            "hu_moment_2": 0,
            "hu_moment_3": 0,
            "hu_moment_4": 0,
            "hu_moment_5": 0,
            "hu_moment_6": 0,
            "hu_moment_7": 0,
            "central_moment_20": 0,
            "central_moment_02": 0,
            "central_moment_11": 0,
        }

    def compute_focus_features_gpu(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """Compute focus features with GPU acceleration"""
        if image is None:
            return self._empty_focus_dict()

        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        masked_image = image * mask

        if not self.use_gpu:
            return self.compute_focus_features_cpu(image, mask)

        img_gpu = self.gpu.to_gpu(masked_image.astype(np.float32))
        mask_gpu = self.gpu.to_gpu(mask.astype(bool))

        if self.gpu.backend == "torch":
            # Sobel gradients using conv2d
            sobel_x = (
                torch.tensor(
                    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                    dtype=torch.float32,
                    device=self.gpu.device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            sobel_y = (
                torch.tensor(
                    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                    dtype=torch.float32,
                    device=self.gpu.device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            img_tensor = img_gpu.unsqueeze(0).unsqueeze(0)
            grad_x = torch.nn.functional.conv2d(
                img_tensor, sobel_x, padding=1
            ).squeeze()
            grad_y = torch.nn.functional.conv2d(
                img_tensor, sobel_y, padding=1
            ).squeeze()

            sobel_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
            sobel_masked = sobel_magnitude[mask_gpu]

            # Laplacian
            laplacian_kernel = (
                torch.tensor(
                    [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                    dtype=torch.float32,
                    device=self.gpu.device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            laplacian = torch.nn.functional.conv2d(
                img_tensor, laplacian_kernel, padding=1
            ).squeeze()
            laplacian_masked = laplacian[mask_gpu]

            # Tenengrad
            tenengrad = grad_x**2 + grad_y**2
            tenengrad_masked = tenengrad[mask_gpu]

            # Intensity stats
            intensity_masked = img_gpu[mask_gpu]

            # Compute statistics
            sobel_stats = self.gpu.compute_stats(sobel_masked)
            laplacian_stats = self.gpu.compute_stats(laplacian_masked)
            tenengrad_stats = self.gpu.compute_stats(tenengrad_masked)
            intensity_stats = self.gpu.compute_stats(intensity_masked)

        elif self.gpu.backend == "cupy":
            # Similar implementation for CuPy
            sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32)
            sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=cp.float32)

            grad_x = ndimage.convolve(
                self.gpu.to_cpu(img_gpu), self.gpu.to_cpu(sobel_x)
            )
            grad_y = ndimage.convolve(
                self.gpu.to_cpu(img_gpu), self.gpu.to_cpu(sobel_y)
            )

            grad_x_gpu = self.gpu.to_gpu(grad_x)
            grad_y_gpu = self.gpu.to_gpu(grad_y)

            sobel_magnitude = cp.sqrt(grad_x_gpu**2 + grad_y_gpu**2)
            sobel_masked = sobel_magnitude[mask_gpu]

            # Continue with CuPy operations...
            laplacian_kernel = cp.array(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=cp.float32
            )
            laplacian = ndimage.convolve(
                self.gpu.to_cpu(img_gpu), self.gpu.to_cpu(laplacian_kernel)
            )
            laplacian_gpu = self.gpu.to_gpu(laplacian)
            laplacian_masked = laplacian_gpu[mask_gpu]

            tenengrad = grad_x_gpu**2 + grad_y_gpu**2
            tenengrad_masked = tenengrad[mask_gpu]
            intensity_masked = img_gpu[mask_gpu]

            sobel_stats = self.gpu.compute_stats(sobel_masked)
            laplacian_stats = self.gpu.compute_stats(laplacian_masked)
            tenengrad_stats = self.gpu.compute_stats(tenengrad_masked)
            intensity_stats = self.gpu.compute_stats(intensity_masked)

        # Brenner gradient (CPU for now due to complexity)
        brenner_gradient = self._compute_brenner_cpu(masked_image, mask)

        # Image entropy (CPU)
        image_entropy = self._compute_entropy_cpu(masked_image, mask)

        # Higher order moments for intensity (CPU for numerical stability)
        intensity_cpu = (
            self.gpu.to_cpu(intensity_masked)
            if self.use_gpu
            else masked_image[mask > 0]
        )
        if len(intensity_cpu) > 0 and intensity_stats["std"] > 0:
            norm_intensity = (
                intensity_cpu - intensity_stats["mean"]
            ) / intensity_stats["std"]
            intensity_skewness = np.mean(norm_intensity**3)
            intensity_kurtosis = np.mean(norm_intensity**4)
        else:
            intensity_skewness = intensity_kurtosis = 0

        return {
            "sobel_mean": sobel_stats["mean"],
            "sobel_std": sobel_stats["std"],
            "laplacian_var": laplacian_stats["var"],
            "edge_strength": sobel_stats["mean"] + sobel_stats["std"],
            "brenner_gradient": brenner_gradient,
            "tenengrad_var": tenengrad_stats["var"],
            "image_entropy": image_entropy,
            "intensity_mean": intensity_stats["mean"],
            "intensity_std": intensity_stats["std"],
            "intensity_skewness": intensity_skewness,
            "intensity_kurtosis": intensity_kurtosis,
        }

    def compute_focus_features_cpu(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """CPU fallback for focus features"""
        if image is None:
            return self._empty_focus_dict()

        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        masked_image = image * mask

        # Sobel gradients
        grad_x = cv2.Sobel(masked_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(masked_image, cv2.CV_64F, 0, 1, ksize=3)

        sobel_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        sobel_masked = sobel_magnitude[mask > 0]

        sobel_mean = np.mean(sobel_masked) if len(sobel_masked) > 0 else 0
        sobel_std = np.std(sobel_masked) if len(sobel_masked) > 0 else 0

        # Laplacian
        laplacian = cv2.Laplacian(masked_image, cv2.CV_64F)
        laplacian_masked = laplacian[mask > 0]
        laplacian_var = np.var(laplacian_masked) if len(laplacian_masked) > 0 else 0

        # Tenengrad
        tenengrad = grad_x**2 + grad_y**2
        tenengrad_masked = tenengrad[mask > 0]
        tenengrad_var = np.var(tenengrad_masked) if len(tenengrad_masked) > 0 else 0

        # Brenner gradient
        brenner_gradient = self._compute_brenner_cpu(masked_image, mask)

        # Image entropy
        image_entropy = self._compute_entropy_cpu(masked_image, mask)

        # Intensity statistics
        intensity_masked = masked_image[mask > 0]
        if len(intensity_masked) > 0:
            intensity_mean = np.mean(intensity_masked)
            intensity_std = np.std(intensity_masked)
            if intensity_std > 0:
                norm_intensity = (intensity_masked - intensity_mean) / intensity_std
                intensity_skewness = np.mean(norm_intensity**3)
                intensity_kurtosis = np.mean(norm_intensity**4)
            else:
                intensity_skewness = intensity_kurtosis = 0
        else:
            intensity_mean = intensity_std = intensity_skewness = intensity_kurtosis = 0

        return {
            "sobel_mean": sobel_mean,
            "sobel_std": sobel_std,
            "laplacian_var": laplacian_var,
            "edge_strength": sobel_mean + sobel_std,
            "brenner_gradient": brenner_gradient,
            "tenengrad_var": tenengrad_var,
            "image_entropy": image_entropy,
            "intensity_mean": intensity_mean,
            "intensity_std": intensity_std,
            "intensity_skewness": intensity_skewness,
            "intensity_kurtosis": intensity_kurtosis,
        }

    def _empty_focus_dict(self):
        """Return empty focus features dict"""
        return {
            "sobel_mean": 0,
            "sobel_std": 0,
            "laplacian_var": 0,
            "edge_strength": 0,
            "brenner_gradient": 0,
            "tenengrad_var": 0,
            "image_entropy": 0,
            "intensity_mean": 0,
            "intensity_std": 0,
            "intensity_skewness": 0,
            "intensity_kurtosis": 0,
        }

    def _compute_brenner_cpu(self, masked_image, mask):
        """Compute Brenner gradient on CPU"""
        if masked_image.shape[0] > 2:
            brenner_kernel = np.array(
                [[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32
            )
            brenner_response = cv2.filter2D(
                masked_image.astype(np.float32), -1, brenner_kernel
            )
            brenner_masked = brenner_response[mask > 0]
            return np.var(brenner_masked) if len(brenner_masked) > 0 else 0
        return 0

    def _compute_entropy_cpu(self, masked_image, mask):
        """Compute image entropy on CPU"""
        intensity_masked = masked_image[mask > 0]
        if len(intensity_masked) > 0:
            hist, _ = np.histogram(intensity_masked, bins=64, range=(0, 256))
            hist = hist[hist > 0]
            if len(hist) > 0:
                prob = hist / np.sum(hist)
                return -np.sum(prob * np.log2(prob))
        return 0

    def compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU with optional GPU acceleration"""
        if self.use_gpu and mask1.size > 1000:  # Only use GPU for larger masks
            mask1_gpu = self.gpu.to_gpu((mask1 > 0).astype(np.float32))
            mask2_gpu = self.gpu.to_gpu((mask2 > 0).astype(np.float32))

            if self.gpu.backend == "torch":
                intersection = torch.sum(mask1_gpu * mask2_gpu).item()
                union = torch.sum(torch.clamp(mask1_gpu + mask2_gpu, 0, 1)).item()
            elif self.gpu.backend == "cupy":
                intersection = float(cp.sum(mask1_gpu * mask2_gpu))
                union = float(cp.sum(cp.clip(mask1_gpu + mask2_gpu, 0, 1)))

            return intersection / union if union > 0 else 0
        else:
            # CPU computation
            intersection = np.sum((mask1 > 0) & (mask2 > 0))
            union = np.sum((mask1 > 0) | (mask2 > 0))
            return intersection / union if union > 0 else 0

    def extract_frame_features(
        self,
        masks_dict: Dict,
        image: Optional[np.ndarray] = None,
        frame_number: int = 0,
    ) -> Dict[int, BlobFeatures]:
        """Extract features for all masks in a single frame with robust error handling"""
        frame_features = {}
        failed_mask_count = 0

        for obj_id, mask_data in masks_dict.items():
            try:
                # Convert RLE to binary mask if needed
                if isinstance(mask_data, dict):
                    if image is not None:
                        h, w = image.shape[:2]
                    else:
                        h, w = 800, 1280  # Default assumption
                    mask = self.rle_to_mask(mask_data, h, w)
                    rle_counts_str = mask_data["counts"]
                else:
                    from pycocotools import mask as mask_utils

                    mask = mask_data.astype(np.uint8)
                    rle = mask_utils.encode(mask)
                    rle_counts_str = rle["counts"]

                # Compute basic features
                basic_features = self.compute_basic_features(mask)
                if basic_features is None:
                    logging.debug(
                        f"Frame {frame_number}, Object {obj_id}: Skipping due to failed basic features"
                    )
                    failed_mask_count += 1
                    continue

                # Filter by area
                if not (self.min_area <= basic_features["area"] <= self.max_area):
                    logging.debug(
                        f"Frame {frame_number}, Object {obj_id}: Skipping due to area filter ({basic_features['area']})"
                    )
                    continue

                # Compute shape features with error handling
                try:
                    shape_features = self.compute_shape_features(mask)
                except Exception as e:
                    logging.warning(
                        f"Frame {frame_number}, Object {obj_id}: Shape features failed: {e}"
                    )
                    shape_features = self._get_default_shape_features()
                    failed_mask_count += 1

                # Compute focus features with error handling
                try:
                    focus_features = self.compute_focus_features(image, mask)
                except Exception as e:
                    logging.warning(
                        f"Frame {frame_number}, Object {obj_id}: Focus features failed: {e}"
                    )
                    focus_features = self._empty_focus_dict()
                    failed_mask_count += 1

                # Debug: Print some values to verify image processing
                if (
                    image is not None and obj_id == list(masks_dict.keys())[0]
                ):  # First object only
                    logging.debug(
                        f"Frame {frame_number}, Obj {obj_id}: "
                        f"Laplacian var: {focus_features['laplacian_var']:.2f}, "
                        f"Sobel mean: {focus_features['sobel_mean']:.2f}, "
                        f"Image shape: {image.shape}"
                    )

                # Determine if blob is in focus
                is_in_focus = focus_features["laplacian_var"] > self.focus_threshold
                focus_score = focus_features["laplacian_var"]

                # Combine all features safely
                try:
                    blob_features = BlobFeatures(
                        rle_counts_str=rle_counts_str,
                        **basic_features,
                        **shape_features,
                        **focus_features,
                        is_in_focus=is_in_focus,
                        focus_score=focus_score,
                    )

                    frame_features[obj_id] = blob_features

                except Exception as e:
                    logging.error(
                        f"Frame {frame_number}, Object {obj_id}: Failed to create BlobFeatures: {e}"
                    )
                    failed_mask_count += 1
                    continue

            except Exception as e:
                logging.error(
                    f"Frame {frame_number}, Object {obj_id}: Critical error processing mask: {e}"
                )
                failed_mask_count += 1
                continue

        # Log summary for this frame
        total_masks = len(masks_dict)
        successful_masks = len(frame_features)

        if failed_mask_count > 0:
            logging.info(
                f"Frame {frame_number}: Processed {successful_masks}/{total_masks} masks successfully, {failed_mask_count} failed"
            )
        else:
            logging.debug(
                f"Frame {frame_number}: Successfully processed all {successful_masks} masks"
            )

        return frame_features

    def detect_redundant_masks(
        self, frame_features: Dict[int, BlobFeatures], masks_dict: Dict
    ) -> List[Tuple[int, int]]:
        """Detect pairs of redundant/overlapping masks"""
        redundant_pairs = []
        obj_ids = list(frame_features.keys())

        for i, obj_id1 in enumerate(obj_ids):
            for j, obj_id2 in enumerate(obj_ids[i + 1 :], i + 1):
                mask1 = masks_dict[obj_id1]
                mask2 = masks_dict[obj_id2]

                # Skip RLE comparison for now
                if isinstance(mask1, dict) or isinstance(mask2, dict):
                    continue

                iou = self.compute_mask_iou(mask1, mask2)

                if iou > self.redundancy_threshold:
                    redundant_pairs.append((obj_id1, obj_id2))

        return redundant_pairs

    def process_single_file(
        self, filepath: str, image_dict: Optional[Dict[int, np.ndarray]] = None
    ) -> pd.DataFrame:
        """Process a single detection file"""
        try:
            detection_data = self.load_detection_file(filepath)
            all_features = []

            for frame_num, masks_dict in detection_data.items():
                image = image_dict.get(frame_num) if image_dict else None

                frame_features = self.extract_frame_features(
                    masks_dict, image, frame_num
                )
                redundant_pairs = self.detect_redundant_masks(
                    frame_features, masks_dict
                )

                # Convert to DataFrame rows
                for obj_id, features in frame_features.items():
                    feature_dict = {
                        "file": str(filepath),
                        "frame": frame_num,
                        "object_id": obj_id,
                        "area": features.area,
                        "centroid_x": features.centroid[0],
                        "centroid_y": features.centroid[1],
                        "bbox_x": features.bbox[0],
                        "bbox_y": features.bbox[1],
                        "bbox_w": features.bbox[2],
                        "bbox_h": features.bbox[3],
                        "aspect_ratio": features.aspect_ratio,
                        "solidity": features.solidity,
                        "extent": features.extent,
                        "perimeter": features.perimeter,
                        "circularity": features.circularity,
                        "eccentricity": features.eccentricity,
                        "major_axis": features.major_axis,
                        "minor_axis": features.minor_axis,
                        "orientation": features.orientation,
                        "roughness": features.roughness,
                        "compactness": features.compactness,
                        "convexity_defect_count": features.convexity_defect_count,
                        "max_defect_depth": features.max_defect_depth,
                        "moment_orientation": features.moment_orientation,
                        "hu_moment_1": features.hu_moment_1,
                        "hu_moment_2": features.hu_moment_2,
                        "hu_moment_3": features.hu_moment_3,
                        "hu_moment_4": features.hu_moment_4,
                        "hu_moment_5": features.hu_moment_5,
                        "hu_moment_6": features.hu_moment_6,
                        "hu_moment_7": features.hu_moment_7,
                        "central_moment_20": features.central_moment_20,
                        "central_moment_02": features.central_moment_02,
                        "central_moment_11": features.central_moment_11,
                        "sobel_mean": features.sobel_mean,
                        "sobel_std": features.sobel_std,
                        "laplacian_var": features.laplacian_var,
                        "edge_strength": features.edge_strength,
                        "brenner_gradient": features.brenner_gradient,
                        "tenengrad_var": features.tenengrad_var,
                        "image_entropy": features.image_entropy,
                        "intensity_mean": features.intensity_mean,
                        "intensity_std": features.intensity_std,
                        "intensity_skewness": features.intensity_skewness,
                        "intensity_kurtosis": features.intensity_kurtosis,
                        "is_in_focus": features.is_in_focus,
                        "focus_score": features.focus_score,
                        "is_redundant": obj_id in [pair[0] for pair in redundant_pairs]
                        or obj_id in [pair[1] for pair in redundant_pairs],
                        "rle_counts_str": features.rle_counts_str,
                    }

                    all_features.append(feature_dict)

            return pd.DataFrame(all_features)

        except Exception as e:
            logging.error(f"Error processing file {filepath}: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def process_video_parallel(
        self,
        file_paths: List[str],
        image_dict: Optional[Dict[int, np.ndarray]] = None,
        use_threads: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> pd.DataFrame:
        """
        Process multiple detection files in parallel

        Args:
            file_paths: List of paths to pkl.gz files
            image_dict: Optional dictionary mapping frame numbers to images
            use_threads: Use ThreadPoolExecutor (True) vs ProcessPoolExecutor (False)
            progress_callback: Optional callback function for progress updates

        Returns:
            Combined DataFrame with all features
        """
        all_dataframes = []
        completed = 0
        total = len(file_paths)

        start_time = time.time()

        if use_threads:
            # Use threads for I/O bound operations
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(
                        self.process_single_file, filepath, image_dict
                    ): filepath
                    for filepath in file_paths
                }

                # Process completed tasks
                for future in as_completed(future_to_file):
                    filepath = future_to_file[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            all_dataframes.append(df)
                        completed += 1

                        if progress_callback:
                            progress_callback(completed, total, filepath)
                        else:
                            elapsed = time.time() - start_time
                            rate = completed / elapsed if elapsed > 0 else 0
                            eta = (total - completed) / rate if rate > 0 else 0
                            print(
                                f"Processed {completed}/{total} files ({completed / total * 100:.1f}%) "
                                f"Rate: {rate:.1f} files/sec ETA: {eta:.1f}s"
                            )

                    except Exception as e:
                        logging.error(f"Error processing {filepath}: {e}")
                        completed += 1
        else:
            # Use processes for CPU-bound operations
            process_func = partial(
                process_file_wrapper,
                analyzer_params={
                    "focus_threshold": self.focus_threshold,
                    "min_area": self.min_area,
                    "max_area": self.max_area,
                    "redundancy_iou_threshold": self.redundancy_threshold,
                    "use_gpu": False,  # Disable GPU in subprocesses
                    "n_workers": 1,
                },
            )

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                future_to_file = {
                    executor.submit(process_func, filepath, image_dict): filepath
                    for filepath in file_paths
                }

                for future in as_completed(future_to_file):
                    filepath = future_to_file[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            all_dataframes.append(df)
                        completed += 1

                        if progress_callback:
                            progress_callback(completed, total, filepath)
                        else:
                            elapsed = time.time() - start_time
                            rate = completed / elapsed if elapsed > 0 else 0
                            eta = (total - completed) / rate if rate > 0 else 0
                            print(
                                f"Processed {completed}/{total} files ({completed / total * 100:.1f}%) "
                                f"Rate: {rate:.1f} files/sec ETA: {eta:.1f}s"
                            )

                    except Exception as e:
                        logging.error(f"Error processing {filepath}: {e}")
                        completed += 1

        # Combine all DataFrames
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            logging.info(
                f"Successfully processed {len(all_dataframes)} files with {len(combined_df)} total detections"
            )
            return combined_df
        else:
            logging.warning("No data processed successfully")
            return pd.DataFrame()

    def process_video_directory(
        self,
        directory_path: str,
        pattern: str = "*.pkl.gz",
        image_directory: Optional[str] = None,
        image_pattern: str = "*.jpg",
        use_threads: bool = True,
    ) -> pd.DataFrame:
        """
        Process all detection files in a directory

        Args:
            directory_path: Path to directory containing pkl.gz files
            pattern: Glob pattern for detection files
            image_directory: Optional path to directory containing images
            image_pattern: Glob pattern for image files
            use_threads: Use ThreadPoolExecutor vs ProcessPoolExecutor

        Returns:
            Combined DataFrame with all features
        """
        directory = Path(directory_path)
        file_paths = list(directory.glob(pattern))
        file_paths.sort()  # Ensure consistent ordering

        logging.info(f"Found {len(file_paths)} detection files to process")

        # Load images if directory provided
        image_dict = None
        if image_directory:
            image_dict = self._load_images_from_directory(
                image_directory, image_pattern
            )
            logging.info(f"Loaded {len(image_dict)} images for processing")

        return self.process_video_parallel(
            [str(p) for p in file_paths], image_dict, use_threads=use_threads
        )

    def _load_images_from_directory(
        self, image_dir: str, pattern: str = "*.jpg"
    ) -> Dict[int, np.ndarray]:
        """
        Load images from directory and map them to frame numbers

        Args:
            image_dir: Path to image directory
            pattern: Glob pattern for image files

        Returns:
            Dictionary mapping frame numbers to images
        """
        image_directory = Path(image_dir)
        image_paths = list(image_directory.glob(pattern))
        image_dict = {}

        logging.info(f"Loading images from {image_dir} with pattern {pattern}")

        for img_path in image_paths:
            try:
                # Extract frame number from filename (your format: XXXX.jpg)
                filename = img_path.stem  # Gets filename without extension

                # Since your files are just numbers (0000, 0001, etc.)
                if filename.isdigit():
                    frame_num = int(filename)
                    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        image_dict[frame_num] = image
                        if len(image_dict) % 100 == 0:  # Progress logging
                            logging.info(f"Loaded {len(image_dict)} images...")
                    else:
                        logging.warning(f"Could not load image: {img_path}")
                else:
                    logging.warning(
                        f"Filename not in expected format (XXXX.jpg): {img_path}"
                    )

            except (ValueError, Exception) as e:
                logging.warning(f"Error processing image {img_path}: {e}")

        if image_dict:
            frame_numbers = sorted(image_dict.keys())
            logging.info(
                f"Successfully loaded {len(image_dict)} images for frames {frame_numbers[0]} to {frame_numbers[-1]}"
            )
        else:
            logging.warning(
                "No images loaded - check filename patterns and directory path"
            )

        return image_dict

    def analyze_video_quality(self, df: pd.DataFrame) -> Dict:
        """
        Analyze overall video quality metrics

        Args:
            df: DataFrame from process_video_parallel or process_video_directory

        Returns:
            Dictionary of quality metrics
        """
        if df.empty:
            return {"error": "No data to analyze"}

        total_detections = len(df)
        in_focus_count = df["is_in_focus"].sum()
        redundant_count = df["is_redundant"].sum()

        # Quality thresholds (you can adjust these)
        quality_blobs = df[
            (df["is_in_focus"] == True)
            & (df["is_redundant"] == False)
            &
            # (df['circularity'] > 0.3) &
            (df["solidity"] > 0.8)
            & (df["laplacian_var"] > self.focus_threshold)
        ]

        frames_analyzed = df["frame"].nunique()
        files_processed = df["file"].nunique()

        # Focus quality distribution
        focus_stats = {
            "mean_laplacian_var": df["laplacian_var"].mean(),
            "median_laplacian_var": df["laplacian_var"].median(),
            "focus_score_std": df["laplacian_var"].std(),
        }

        # Shape quality distribution
        shape_stats = {
            "mean_circularity": df["circularity"].mean(),
            "mean_solidity": df["solidity"].mean(),
            "mean_area": df["area"].mean(),
        }

        return {
            "total_detections": total_detections,
            "quality_detections": len(quality_blobs),
            "quality_percentage": len(quality_blobs) / total_detections * 100,
            "in_focus_percentage": in_focus_count / total_detections * 100,
            "redundant_percentage": redundant_count / total_detections * 100,
            "frames_analyzed": frames_analyzed,
            "files_processed": files_processed,
            "focus_quality": focus_stats,
            "shape_quality": shape_stats,
            "detections_per_frame": total_detections / frames_analyzed
            if frames_analyzed > 0
            else 0,
        }


# Helper function for multiprocessing
def process_file_wrapper(
    filepath: str, image_dict: Optional[Dict] = None, analyzer_params: Dict = None
) -> pd.DataFrame:
    """Wrapper function for multiprocessing"""
    analyzer = SAM2BlobAnalyzer(**analyzer_params)
    return analyzer.process_single_file(filepath, image_dict)


# Batch processing utilities
class VideoProcessor:
    """High-level interface for batch video processing"""

    def __init__(self, analyzer: SAM2BlobAnalyzer):
        self.analyzer = analyzer

    def _load_images_from_directory(
        self, image_dir: str, pattern: str = "*.jpg"
    ) -> Dict[int, np.ndarray]:
        """
        Load images from directory and map them to frame numbers

        Args:
            image_dir: Path to image directory
            pattern: Glob pattern for image files

        Returns:
            Dictionary mapping frame numbers to images
        """
        image_directory = Path(image_dir)
        image_paths = list(image_directory.glob(pattern))
        image_dict = {}

        logging.info(f"Loading images from {image_dir} with pattern {pattern}")

        for img_path in image_paths:
            try:
                # Extract frame number from filename (your format: XXXX.jpg)
                filename = img_path.stem  # Gets filename without extension

                # Since your files are just numbers (0000, 0001, etc.)
                if filename.isdigit():
                    frame_num = int(filename)
                    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        image_dict[frame_num] = image
                        if len(image_dict) % 100 == 0:  # Progress logging
                            logging.info(f"Loaded {len(image_dict)} images...")
                    else:
                        logging.warning(f"Could not load image: {img_path}")
                else:
                    logging.warning(
                        f"Filename not in expected format (XXXX.jpg): {img_path}"
                    )

            except (ValueError, Exception) as e:
                logging.warning(f"Error processing image {img_path}: {e}")

        if image_dict:
            frame_numbers = sorted(image_dict.keys())
            logging.info(
                f"Successfully loaded {len(image_dict)} images for frames {frame_numbers[0]} to {frame_numbers[-1]}"
            )
        else:
            logging.warning(
                "No images loaded - check filename patterns and directory path"
            )

        return image_dict

    def process_video_sequence(
        self,
        detection_dir: str,
        output_path: str,
        image_dir: Optional[str] = None,
        image_pattern: str = "*.jpg",
        chunk_size: int = 100,
        save_intermediate: bool = True,
    ) -> pd.DataFrame:
        """
        Process large video sequences in chunks to manage memory

        Args:
            detection_dir: Directory containing pkl.gz files
            output_path: Path to save final results
            image_dir: Optional directory containing images
            image_pattern: Glob pattern for image files
            chunk_size: Number of files to process per chunk
            save_intermediate: Whether to save intermediate results

        Returns:
            Final combined DataFrame
        """
        directory = Path(detection_dir)
        file_paths = sorted(list(directory.glob("*.pkl.gz")))

        total_files = len(file_paths)
        logging.info(f"Processing {total_files} files in chunks of {chunk_size}")

        # Load images if directory provided
        image_dict = None
        if image_dir:
            image_dict = self._load_images_from_directory(image_dir, image_pattern)
            logging.info(f"Loaded {len(image_dict)} images for processing")

        all_chunks = []

        for i in range(0, total_files, chunk_size):
            chunk_files = file_paths[i : i + chunk_size]
            chunk_num = i // chunk_size + 1
            total_chunks = (total_files + chunk_size - 1) // chunk_size

            logging.info(
                f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk_files)} files)"
            )

            # Process chunk with images
            chunk_df = self.analyzer.process_video_parallel(
                [str(p) for p in chunk_files],
                image_dict=image_dict,  # Pass the loaded images
                use_threads=True,
            )

            if not chunk_df.empty:
                all_chunks.append(chunk_df)

                if save_intermediate:
                    chunk_path = f"{output_path}_chunk_{chunk_num:03d}.parquet"
                    chunk_df.to_parquet(chunk_path)
                    logging.info(f"Saved intermediate chunk to {chunk_path}")

        # Combine all chunks
        if all_chunks:
            final_df = pd.concat(all_chunks, ignore_index=True)

            # Save final result
            if output_path.endswith(".parquet"):
                final_df.to_parquet(output_path)
            elif output_path.endswith(".csv"):
                final_df.to_csv(output_path, index=False)
            else:
                final_df.to_parquet(f"{output_path}.parquet")

            logging.info(
                f"Saved final results with {len(final_df)} detections to {output_path}"
            )
            return final_df
        else:
            logging.error("No data processed successfully")
            return pd.DataFrame()


# Example usage and performance optimization
if __name__ == "__main__":
    import os

    from directory_dict import directory_dict, get_mask_and_frame_and_parent_dirs

    # parser = argparse.ArgumentParser(description="Process SAM2 video detections")
    # parser.add_argument("input_dir", help="Directory containing pkl.gz files")
    # parser.add_argument("output_path", help="Output path for results")
    # parser.add_argument("--image_dir", help="Directory containing images")
    # parser.add_argument(
    #     "--gpu_backend", choices=["torch", "cupy", "auto"], default="torch"
    # )
    # parser.add_argument(
    #     "--workers", type=int, default=None, help="Number of parallel workers"
    # )
    # parser.add_argument("--chunk_size", type=int, default=100, help="Files per chunk")
    # parser.add_argument("--focus_threshold", type=float, default=100.0)
    # parser.add_argument(
    #     "--use_processes", action="store_true", help="Use processes instead of threads"
    # )

    # args = parser.parse_args()

    #     # Initialize analyzer
    # analyzer = SAM2BlobAnalyzer(
    #     focus_threshold=args.focus_threshold,
    #     use_gpu=True,
    #     gpu_backend=args.gpu_backend,
    #     n_workers=args.workers,
    # )

    # # Process video
    # processor = VideoProcessor(analyzer)

    # start_time = time.time()
    # results_df = processor.process_video_sequence(
    #     detection_dir=args.input_dir,
    #     output_path=args.output_path,
    #     image_dir=args.image_dir,
    #     chunk_size=args.chunk_size,
    # )

    vid_key = "vid64_rolling5"
    input_dir, image_dir, parent_dir = get_mask_and_frame_and_parent_dirs(
        directory_dict, vid_key
    )
    output_path = os.path.join(parent_dir, "results.parquet")
    gpu_backend = "torch"
    workers = 30
    chunk_size = 2000
    focus_threshold = 1000.0

    # Initialize analyzer
    analyzer = SAM2BlobAnalyzer(
        focus_threshold=focus_threshold,
        use_gpu=False,
        gpu_backend=gpu_backend,
        n_workers=workers,
    )

    # Process video
    processor = VideoProcessor(analyzer)

    start_time = time.time()
    results_df = processor.process_video_sequence(
        detection_dir=input_dir,
        output_path=output_path,
        image_dir=image_dir,
        chunk_size=chunk_size,
    )

    processing_time = time.time() - start_time

    # Analyze results
    quality_metrics = analyzer.analyze_video_quality(results_df)

    print("\n=== Processing Complete ===")
    print(f"Processing time: {processing_time:.1f} seconds")
    print(f"Total detections: {quality_metrics['total_detections']}")
    print(
        f"Quality detections: {quality_metrics['quality_detections']} ({quality_metrics['quality_percentage']:.1f}%)"
    )
    print(f"In-focus percentage: {quality_metrics['in_focus_percentage']:.1f}%")
    print(f"Redundant percentage: {quality_metrics['redundant_percentage']:.1f}%")
    print(f"Frames analyzed: {quality_metrics['frames_analyzed']}")
    print(
        f"Average detections per frame: {quality_metrics['detections_per_frame']:.1f}"
    )

    # # Example filtering for high-quality blobs
    # quality_blobs = results_df[
    #     (results_df["is_in_focus"] == True)
    #     & (results_df["is_redundant"] == False)
    #     # & (results_df["circularity"] > 0.3)
    #     & (results_df["solidity"] > 0.8)
    # ]

    # if len(quality_blobs) > 0:
    #     quality_output = output_path.replace(".parquet", "_quality.parquet").replace(
    #         ".csv", "_quality.csv"
    #     )
    #     if quality_output.endswith(".parquet"):
    #         quality_blobs.to_parquet(quality_output)
    #     else:
    #         quality_blobs.to_csv(quality_output, index=False)
    #     print(f"Saved {len(quality_blobs)} quality blobs to {quality_output}")
