"""
Enhanced mask video generator with parquet support and flexible visualization options.

Supports:
- Loading from pickle.gz files (original behavior)
- Loading from parquet files with RLE-encoded masks
- Color mapping by trajectory ID (default) or any numeric/categorical variable
- Annotations (frame numbers, scale bars, labels, statistics)
- Quality indicators (outlines, opacity based on metrics)
- GPU acceleration with graceful CPU fallback
"""

import gzip
import hashlib
import os
import pickle
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from pycocotools import mask as mask_utils
from tqdm import tqdm

try:
    import cupy as cp
    import torch

    TORCH_AVAILABLE = True
    CUPY_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    CUPY_AVAILABLE = False
    warnings.warn("PyTorch/CuPy not available, falling back to CPU processing")


@dataclass
class VisualizationConfig:
    """Configuration for video visualization."""

    # Basic settings
    alpha: float = 0.5
    fps: int = 30
    output_codec: str = "mp4v"  # 'mp4v', 'avc1', 'XVID'

    # Coloring
    color_by: Optional[str] = None  # None = trajectory_id, or column name from parquet
    colormap: str = "viridis"  # for continuous variables
    vmin: Optional[float] = None
    vmax: Optional[float] = None

    # Annotations
    show_frame_number: bool = False
    show_scale_bar: bool = False
    scale_bar_length_um: float = 100.0
    scale_bar_position: Tuple[int, int] = (50, 50)  # pixels from bottom-left
    pixels_per_um: float = 2.5  # for scale bar conversion

    show_trajectory_labels: bool = False
    label_font_scale: float = 0.4
    label_thickness: int = 1

    # Quality indicators
    outline_mode: Optional[Literal["all", "filtered", "confidence"]] = None
    outline_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    outline_thickness: int = 2
    confidence_threshold: float = 0.5  # for outline_mode="confidence"

    # Temporal effects
    fade_trail_frames: int = 0  # Show fading trail of last N frames
    trail_alpha_decay: float = 0.7  # Multiplicative decay per frame

    # Statistics overlay
    show_stats: bool = False
    stats_position: Tuple[int, int] = (10, 30)  # pixels from top-left

    # Filtering
    filter_column: Optional[str] = None  # Column to filter by
    filter_values: Optional[Set[Any]] = None  # Values to include

    def __post_init__(self):
        """Validate configuration."""
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.fade_trail_frames < 0:
            raise ValueError("fade_trail_frames must be non-negative")


class MaskDataLoader:
    """Handles loading mask data from various sources."""

    @staticmethod
    def from_pickle_dir(
        mask_dir: str, frame_nums: Optional[List[int]] = None
    ) -> Dict[int, Dict[int, Any]]:
        """Load masks from directory of .pkl.gz files."""
        if frame_nums is None:
            frame_nums = sorted(
                int(f.split(".")[0])
                for f in os.listdir(mask_dir)
                if f.endswith(".pkl.gz")
            )

        mask_dict = {}
        for frame_num in tqdm(frame_nums, desc="Loading pickle masks"):
            fname = f"{frame_num:04d}.pkl.gz"
            path = os.path.join(mask_dir, fname)
            if not os.path.exists(path):
                continue

            with gzip.open(path, "rb") as f:
                frame_data = pickle.load(f)
                # Handle nested structure {frame_idx: {obj_id: rle}}
                if len(frame_data) == 1 and isinstance(
                    list(frame_data.values())[0], dict
                ):
                    frame_data = list(frame_data.values())[0]
                mask_dict[frame_num] = frame_data

        return mask_dict

    @staticmethod
    def from_parquet(
        parquet_path: str,
        rle_column: str = "rle_counts_str",
        frame_column: str = "frame",
        obj_id_column: str = "trajectory_id",
        metadata_columns: Optional[List[str]] = None,
    ) -> Tuple[Dict[int, Dict[int, Any]], Optional[pd.DataFrame]]:
        """
        Load masks from parquet file with RLE-encoded strings.

        Returns:
            mask_dict: {frame_num: {obj_id: rle_dict}}
            metadata_df: DataFrame with all columns for each (frame, obj_id) pair
        """
        df = pd.read_parquet(parquet_path)

        required_cols = [frame_column, obj_id_column, rle_column]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Parquet missing required columns: {missing}")

        mask_dict = defaultdict(dict)

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading parquet masks"):
            frame_num = int(row[frame_column])
            obj_id = int(row[obj_id_column])
            rle_str = row[rle_column]

            # Convert RLE string to pycocotools format
            # Assuming format is compressed RLE string from pycocotools
            rle_dict = {
                "counts": rle_str.encode() if isinstance(rle_str, str) else rle_str,
                "size": [int(row.get("height", 800)), int(row.get("width", 1280))],
            }
            mask_dict[frame_num][obj_id] = rle_dict

        # Keep metadata if requested
        if metadata_columns is not None:
            keep_cols = [frame_column, obj_id_column] + metadata_columns
            metadata_df = df[[c for c in keep_cols if c in df.columns]].copy()
        else:
            metadata_df = df.copy()

        return dict(mask_dict), metadata_df

    @staticmethod
    def merge_video_segments(directory: str) -> Dict[int, Dict[int, Any]]:
        """Load and merge multiple video_segments_rle.pkl.gz files."""
        consolidated = defaultdict(dict)
        files = sorted(
            f for f in os.listdir(directory) if f.endswith("_video_segments_rle.pkl.gz")
        )

        for filename in tqdm(files, desc="Merging video segments"):
            path = os.path.join(directory, filename)
            with gzip.open(path, "rb") as f:
                data = pickle.load(f)
            for frame_idx, obj_dict in data.items():
                consolidated[frame_idx].update(obj_dict)

        return dict(consolidated)


class ColorMapper:
    """Handles color mapping for different visualization modes."""

    def __init__(
        self,
        mode: str = "trajectory_id",
        colormap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        metadata_df: Optional[pd.DataFrame] = None,
    ):
        self.mode = mode
        self.colormap_name = colormap
        self.vmin = vmin
        self.vmax = vmax
        self.metadata_df = metadata_df
        self._color_cache = {}

        if mode != "trajectory_id" and metadata_df is None:
            raise ValueError("metadata_df required for non-trajectory_id coloring")

        # Setup colormap for continuous variables
        if metadata_df is not None and mode in metadata_df.columns:
            values = metadata_df[mode].dropna()
            if pd.api.types.is_numeric_dtype(values):
                self.is_continuous = True
                self.vmin = vmin if vmin is not None else values.min()
                self.vmax = vmax if vmax is not None else values.max()
                self.normalizer = Normalize(vmin=self.vmin, vmax=self.vmax)
                self.cmap = cm.get_cmap(colormap)
            else:
                self.is_continuous = False
                unique_vals = values.unique()
                self.category_map = {
                    val: self._deterministic_color(val) for val in unique_vals
                }
        else:
            self.is_continuous = False

    def get_color(
        self, obj_id: int, frame_num: Optional[int] = None
    ) -> Tuple[float, float, float]:
        """Get RGB color for given object ID."""
        cache_key = (obj_id, frame_num)
        if cache_key in self._color_cache:
            return self._color_cache[cache_key]

        if self.mode == "trajectory_id":
            color = self._deterministic_color(obj_id)
        elif self.metadata_df is not None:
            # Lookup value from metadata
            mask = self.metadata_df["trajectory_id"] == obj_id
            if frame_num is not None:
                mask &= self.metadata_df["frame"] == frame_num

            if mask.any():
                value = self.metadata_df.loc[mask, self.mode].iloc[0]
                if self.is_continuous:
                    color = self.cmap(self.normalizer(value))[:3]
                else:
                    color = self.category_map.get(value, (0.5, 0.5, 0.5))
            else:
                color = (0.5, 0.5, 0.5)  # Default gray
        else:
            color = self._deterministic_color(obj_id)

        self._color_cache[cache_key] = color
        return color

    @staticmethod
    def _deterministic_color(obj_id: Union[int, Any]) -> Tuple[float, float, float]:
        """Generate deterministic color from ID using hash."""
        h = hashlib.md5(str(obj_id).encode()).hexdigest()
        return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def decode_rle_mask(rle: Any, shape: Tuple[int, int]) -> np.ndarray:
    """Decode RLE mask to binary array."""
    if isinstance(rle, dict):
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, *shape)
        return mask_utils.decode(rle).astype(np.uint8)
    elif isinstance(rle, np.ndarray):
        return rle.squeeze().astype(np.uint8)
    else:
        raise ValueError(f"Unsupported RLE format: {type(rle)}")


def generate_frame_rgba_cpu(
    frame_obj_dict: Dict[int, Any],
    shape: Tuple[int, int],
    color_mapper: ColorMapper,
    alpha: float = 0.5,
    frame_num: Optional[int] = None,
) -> np.ndarray:
    """Generate RGBA composite for a frame using NumPy (CPU)."""
    height, width = shape
    rgba = np.zeros((height, width, 4), dtype=np.float32)

    for obj_id, rle in frame_obj_dict.items():
        mask = decode_rle_mask(rle, (height, width))
        color = color_mapper.get_color(obj_id, frame_num)
        rgba[..., :3] += mask[..., None] * color
        rgba[..., 3] += mask * alpha

    rgba[..., :3] = np.clip(rgba[..., :3], 0, 1)
    rgba[..., 3] = np.clip(rgba[..., 3], 0, 1)
    return rgba


def generate_frame_rgba_gpu(
    frame_obj_dict: Dict[int, Any],
    shape: Tuple[int, int],
    color_mapper: ColorMapper,
    alpha: float = 0.5,
    device: str = "cuda",
    frame_num: Optional[int] = None,
) -> np.ndarray:
    """Generate RGBA composite for a frame using PyTorch (GPU)."""
    if not TORCH_AVAILABLE:
        return generate_frame_rgba_cpu(
            frame_obj_dict, shape, color_mapper, alpha, frame_num
        )

    height, width = shape
    rgba = torch.zeros((4, height, width), dtype=torch.float32, device=device)

    for obj_id, rle in frame_obj_dict.items():
        mask_np = decode_rle_mask(rle, (height, width))
        mask = torch.from_numpy(mask_np).float().to(device)
        color = torch.tensor(
            color_mapper.get_color(obj_id, frame_num),
            dtype=torch.float32,
            device=device,
        ).view(3, 1, 1)

        rgba[:3] += mask * color
        rgba[3] += mask * alpha

    rgba[:3] = rgba[:3].clamp(0, 1)
    rgba[3] = rgba[3].clamp(0, 1)
    return rgba.permute(1, 2, 0).contiguous().cpu().numpy()


def add_annotations(
    frame: np.ndarray, frame_num: int, config: VisualizationConfig, num_objects: int
) -> np.ndarray:
    """Add annotations to frame (scale bar, frame number, etc.)."""
    annotated = (frame * 255).astype(np.uint8)

    # Frame number
    if config.show_frame_number:
        cv2.putText(
            annotated,
            f"Frame: {frame_num}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    # Scale bar
    if config.show_scale_bar:
        height = annotated.shape[0]
        x, y = config.scale_bar_position
        bar_length_px = int(config.scale_bar_length_um * config.pixels_per_um)

        # Draw bar
        cv2.line(
            annotated,
            (x, height - y),
            (x + bar_length_px, height - y),
            (255, 255, 255),
            3,
        )
        # Draw end caps
        cv2.line(
            annotated, (x, height - y - 5), (x, height - y + 5), (255, 255, 255), 2
        )
        cv2.line(
            annotated,
            (x + bar_length_px, height - y - 5),
            (x + bar_length_px, height - y + 5),
            (255, 255, 255),
            2,
        )
        # Label
        cv2.putText(
            annotated,
            f"{config.scale_bar_length_um:.0f} um",
            (x, height - y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    # Statistics
    if config.show_stats:
        x, y = config.stats_position
        cv2.putText(
            annotated,
            f"Droplets: {num_objects}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return annotated.astype(np.float32) / 255.0


def add_outlines(
    frame: np.ndarray,
    frame_obj_dict: Dict[int, Any],
    shape: Tuple[int, int],
    config: VisualizationConfig,
) -> np.ndarray:
    """Add outlines around masks."""
    height, width = shape
    outline_frame = frame.copy()

    for obj_id, rle in frame_obj_dict.items():
        mask = decode_rle_mask(rle, (height, width))

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw outlines
        for contour in contours:
            cv2.drawContours(
                (outline_frame * 255).astype(np.uint8),
                [contour],
                -1,
                tuple(int(c * 255) for c in config.outline_color),
                config.outline_thickness,
            )

    return outline_frame


def preload_images(
    image_folder: str, frame_nums: List[int], shape: Tuple[int, int]
) -> Dict[int, np.ndarray]:
    """Load and resize all images for overlay."""
    height, width = shape
    images = {}

    for frame_num in tqdm(frame_nums, desc="Preloading images"):
        img_path = os.path.join(image_folder, f"{frame_num:04d}.jpg")
        if not os.path.exists(img_path):
            warnings.warn(f"Missing image: {img_path}")
            continue

        bgr = cv2.imread(img_path)
        if bgr is None:
            warnings.warn(f"Failed to load: {img_path}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (height, width):
            rgb = cv2.resize(rgb, (width, height))
        images[frame_num] = rgb.astype(np.float32) / 255.0

    return images


def infer_shape(
    mask_dict: Dict[int, Dict[int, Any]], image_folder: Optional[str] = None
) -> Tuple[int, int]:
    """Infer frame shape from masks or images."""
    # Try from masks
    for frame_data in mask_dict.values():
        for rle in frame_data.values():
            if isinstance(rle, dict) and "size" in rle:
                return tuple(rle["size"])

    # Try from images
    if image_folder:
        sample_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
        if sample_files:
            img = cv2.imread(os.path.join(image_folder, sample_files[0]))
            if img is not None:
                return img.shape[:2]

    raise ValueError("Unable to infer shape from masks or images")


def write_video_from_masks(
    mask_dict: Dict[int, Dict[int, Any]],
    output_path: str,
    image_folder: Optional[str] = None,
    shape: Optional[Tuple[int, int]] = None,
    config: Optional[VisualizationConfig] = None,
    metadata_df: Optional[pd.DataFrame] = None,
    use_gpu: bool = True,
    device: str = "cuda",
) -> None:
    """
    Write video from mask dictionary with flexible visualization options.

    Args:
        mask_dict: {frame_num: {obj_id: rle_mask}}
        output_path: Path for output video
        image_folder: Optional folder with background images
        shape: (height, width) for output video
        config: VisualizationConfig instance
        metadata_df: DataFrame with metadata for color mapping
        use_gpu: Use GPU acceleration if available
        device: 'cuda' or 'cpu'
    """
    if config is None:
        config = VisualizationConfig()

    # Validate GPU availability
    if use_gpu and not (TORCH_AVAILABLE or CUPY_AVAILABLE):
        warnings.warn("GPU requested but not available, using CPU")
        use_gpu = False

    # Infer shape if not provided
    if shape is None:
        shape = infer_shape(mask_dict, image_folder)
    height, width = shape

    # Setup color mapper
    color_mapper = ColorMapper(
        mode=config.color_by or "trajectory_id",
        colormap=config.colormap,
        vmin=config.vmin,
        vmax=config.vmax,
        metadata_df=metadata_df,
    )

    # Get frame numbers
    frame_nums = sorted(mask_dict.keys())

    # Preload images if provided
    images = None
    if image_folder:
        images = preload_images(image_folder, frame_nums, shape)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*config.output_codec)
    writer = cv2.VideoWriter(output_path, fourcc, config.fps, (width, height))

    # Trail buffer for temporal effects
    trail_buffer = []

    # Process frames
    for frame_num in tqdm(frame_nums, desc="Writing video"):
        frame_masks = mask_dict[frame_num]

        # Filter if requested
        if config.filter_column and config.filter_values:
            if metadata_df is not None:
                mask = metadata_df["frame"] == frame_num
                mask &= metadata_df[config.filter_column].isin(config.filter_values)
                valid_ids = set(metadata_df.loc[mask, "trajectory_id"])
                frame_masks = {k: v for k, v in frame_masks.items() if k in valid_ids}

        # Generate RGBA overlay
        if use_gpu:
            rgba = generate_frame_rgba_gpu(
                frame_masks, shape, color_mapper, config.alpha, device, frame_num
            )
        else:
            rgba = generate_frame_rgba_cpu(
                frame_masks, shape, color_mapper, config.alpha, frame_num
            )

        # Apply trail effect
        if config.fade_trail_frames > 0:
            trail_buffer.append(rgba.copy())
            if len(trail_buffer) > config.fade_trail_frames:
                trail_buffer.pop(0)

            # Composite trails
            trail_composite = np.zeros_like(rgba)
            for i, trail_rgba in enumerate(trail_buffer[:-1]):
                decay = config.trail_alpha_decay ** (len(trail_buffer) - i - 1)
                trail_composite += trail_rgba * decay
            trail_composite += rgba
            rgba = np.clip(trail_composite, 0, 1)

        # Blend with background
        if images and frame_num in images:
            bg = images[frame_num]
            blended = rgba[..., :3] * rgba[..., 3:4] + bg * (1 - rgba[..., 3:4])
        else:
            blended = rgba[..., :3]

        # Add outlines
        if config.outline_mode:
            blended = add_outlines(blended, frame_masks, shape, config)

        # Add annotations
        blended = add_annotations(blended, frame_num, config, len(frame_masks))

        # Convert to BGR and write
        bgr_out = cv2.cvtColor((blended * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(bgr_out)

    writer.release()
    print(f"✅ Video written to {output_path}")
    print(f"📊 Processed {len(frame_nums)} frames at {config.fps} FPS")


def main():
    """Example usage."""
    # Example 1: Load from pickle directory
    mask_dict = MaskDataLoader.from_pickle_dir("path/to/masks")

    # Example 2: Load from parquet
    mask_dict, metadata_df = MaskDataLoader.from_parquet(
        "path/to/filtered_droplets.parquet",
        metadata_columns=["area", "mean_curvature", "quality_score"],
    )

    # Example 3: Create video with default settings
    config = VisualizationConfig(
        alpha=0.5,
        fps=100,
        show_frame_number=True,
        show_scale_bar=True,
        scale_bar_length_um=100.0,
        pixels_per_um=2.41,
    )

    write_video_from_masks(
        mask_dict=mask_dict,
        output_path="output.mp4",
        image_folder="path/to/frames",
        config=config,
        use_gpu=True,
    )

    # Example 4: Color by curvature
    config = VisualizationConfig(
        color_by="mean_curvature",
        colormap="RdYlBu_r",
        vmin=-0.01,
        vmax=0.01,
        show_stats=True,
    )

    write_video_from_masks(
        mask_dict=mask_dict,
        output_path="output_curvature.mp4",
        image_folder="path/to/frames",
        config=config,
        metadata_df=metadata_df,
        use_gpu=True,
    )


if __name__ == "__main__":
    # main()
    from pathlib import Path

    # parquet_file = "/home/leviathan/cdrucker/ml/sam2/outputs/vid363_rolling1/blob_analysis/blob_features.parquet"
    # frame_dir = "/home/leviathan/cdrucker/videos/jpegVideoFrames/363_0p8mlmin_800fps_lambda5_500prep_1000cent"
    # mask_dict, metadata_df = MaskDataLoader.from_parquet(
    #     parquet_path=parquet_file,
    #     metadata_columns=["area", "perimeter", "aspect_ratio"],
    #     obj_id_column="object_id",
    # )

    # config = VisualizationConfig(
    #     alpha=0.5,
    #     fps=100,
    #     show_frame_number=True,
    #     show_scale_bar=True,
    #     scale_bar_length_um=100.0,
    #     pixels_per_um=0.41,
    # )

    # write_video_from_masks(
    #     mask_dict=mask_dict,
    #     output_path=Path(parquet_file).parent / "animation_100fps.mp4",
    #     image_folder=frame_dir,
    #     config=config,
    #     use_gpu=True,
    # )

    parquet_files = [
        "/home/leviathan/cdrucker/ml/sam2/outputs/vid64_rolling8/blob_analysis/blob_features.parquet"
    ]
    frame_dirs = [
        "/home/leviathan/cdrucker/videos/jpegVideoFrames/17758_1_64_0p2mlmin1_lambda5_2krpm_400fps"
    ]

    for ind in range(len(parquet_files)):
        parquet_file = parquet_files[ind]
        frame_dir = frame_dirs[ind]
        mask_dict, metadata_df = MaskDataLoader.from_parquet(
            parquet_path=parquet_file,
            metadata_columns=["area", "perimeter", "aspect_ratio"],
            obj_id_column="object_id",
        )

        config = VisualizationConfig(
            alpha=0.5,
            fps=100,
            show_frame_number=True,
            show_scale_bar=True,
            scale_bar_length_um=100.0,
            pixels_per_um=0.41,
        )
        write_video_from_masks(
            mask_dict=mask_dict,
            output_path=Path(parquet_file).parent / "animation_100fps.mp4",
            image_folder=frame_dir,
            config=config,
            use_gpu=True,
        )
    # config = VisualizationConfig(
    #     color_by="area",
    #     colormap="RdYlBu_r",
    #     show_stats=True,
    #     fps=100
    # )

    # write_video_from_masks(
    #     mask_dict=mask_dict,
    #     output_path=Path(parquet_file).parent / "output_area.mp4",
    #     image_folder=frame_dir,
    #     config=config,
    #     metadata_df=metadata_df,
    #     use_gpu=True,
    # )
