import gzip
import hashlib
import os
import pickle
import re
import time
from collections import defaultdict
from typing import Any, Dict

import cv2
import numpy as np
import torch
from matplotlib import cm
from pycocotools import mask as mask_utils
from tqdm import tqdm


def decode_rle_mask(rle, shape):
    """Decodes an RLE mask using pycocotools."""
    if isinstance(rle, dict) and isinstance(rle.get("counts"), list):
        rle = mask_utils.frPyObjects(rle, *shape)
    elif isinstance(rle, np.ndarray):
        return rle.squeeze().astype(np.uint8)
    return mask_utils.decode(rle).astype(np.uint8)


def generate_color_map(object_ids):
    """Assigns a distinct color to each object ID."""
    cmap = cm.get_cmap("tab20", len(object_ids))
    return {obj_id: cmap(i)[:3] for i, obj_id in enumerate(object_ids)}


def generate_frame_rgba(frame_obj_dict, shape, color_map, alpha=0.5):
    """Generates an RGBA composite for a single frame using NumPy (CPU)."""
    height, width = shape
    rgba = np.zeros((height, width, 4), dtype=np.float32)

    for obj_id, rle in frame_obj_dict.items():
        mask = decode_rle_mask(rle, (height, width))
        color = color_map[obj_id]
        rgba[..., :3] += mask[..., None] * color
        rgba[..., 3] += mask * alpha

    rgba[..., :3] = np.clip(rgba[..., :3], 0, 1)
    rgba[..., 3] = np.clip(rgba[..., 3], 0, 1)
    return rgba


def generate_frame_rgba_torch(
    frame_obj_dict, shape, color_map, alpha=0.5, device="cuda"
):
    """Generates an RGBA composite for a single frame using PyTorch (GPU)."""
    height, width = shape
    rgba = torch.zeros((4, height, width), dtype=torch.float32, device=device)

    for obj_id, rle in frame_obj_dict.items():
        mask_np = decode_rle_mask(rle, (height, width))
        mask = torch.from_numpy(mask_np).float().to(device)
        # color = torch.tensor(
        #     color_map[obj_id], dtype=torch.float32, device=device
        # ).view(3, 1, 1)
        color = torch.tensor(
            get_or_assign_color(color_map, obj_id), dtype=torch.float32, device=device
        ).view(3, 1, 1)
        rgba[:3] += mask * color
        rgba[3] += mask * alpha

    rgba[:3] = rgba[:3].clamp(0, 1)
    rgba[3] = rgba[3].clamp(0, 1)
    rgba = rgba.permute(1, 2, 0).contiguous().cpu().numpy()
    return rgba


def preload_images(image_folder, frame_nums, shape):
    """Loads and resizes all images needed for overlay."""
    height, width = shape
    images = {}
    for frame_num in tqdm(frame_nums, desc="Preloading images"):
        img_path = os.path.join(image_folder, f"{frame_num:04d}.jpg")
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Missing image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (width, height))
        images[frame_num] = rgb.astype(np.float32) / 255.0
    return images


def infer_shape_from_data(mask_dict, image_folder=None):
    """Infers frame shape from first available RLE or image."""
    for frame_data in mask_dict.values():
        for rle in frame_data.values():
            if isinstance(rle, dict) and "size" in rle:
                return tuple(rle["size"])
    if image_folder:
        sample_path = next(
            (
                os.path.join(image_folder, f)
                for f in os.listdir(image_folder)
                if f.endswith(".jpg")
            ),
            None,
        )
        if sample_path:
            img = cv2.imread(sample_path)
            if img is not None:
                return img.shape[:2]
    raise ValueError("Unable to infer shape from data or image folder")


def write_video_from_masks(
    mask_dict,
    output_path,
    image_folder=None,
    fps=30,
    alpha=0.5,
    shape=None,
    use_torch=False,
    device="cuda",
):
    """Writes a video visualizing the RLE mask dict over optional images."""
    overall_start = time.time()

    frame_nums = sorted(mask_dict.keys())
    all_obj_ids = {obj_id for frame in mask_dict.values() for obj_id in frame}
    color_map = generate_color_map(sorted(all_obj_ids))

    if shape is None:
        shape = infer_shape_from_data(mask_dict, image_folder)

    preload_start = time.time()
    images = preload_images(image_folder, frame_nums, shape) if image_folder else None
    preload_duration = time.time() - preload_start

    height, width = shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    process_times = []

    for frame_num in tqdm(frame_nums, desc="Writing video"):
        frame_start = time.time()

        frame_masks = mask_dict[frame_num]
        if use_torch:
            rgba = generate_frame_rgba_torch(
                frame_masks, shape, color_map, alpha, device=device
            )
        else:
            rgba = generate_frame_rgba(frame_masks, shape, color_map, alpha)

        if images:
            bg = images[frame_num]
            blended = rgba[..., :3] * rgba[..., 3:4] + bg * (1 - rgba[..., 3:4])
        else:
            blended = rgba[..., :3]

        bgr_out = cv2.cvtColor((blended * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(bgr_out)

        process_times.append(time.time() - frame_start)

    writer.release()
    overall_duration = time.time() - overall_start

    total_frames = len(frame_nums)
    avg_frame_time = np.mean(process_times)
    avg_fps = total_frames / overall_duration

    print(f"✅ Video written to {output_path}")
    print(
        f"📊 Processed {total_frames} frames in {overall_duration:.2f} seconds ({avg_fps:.2f} FPS)"
    )
    print(f"📂 Image preload took {preload_duration:.2f} seconds")
    print(f"🧠 Average processing time per frame: {avg_frame_time:.3f} seconds")


def load_mask_for_frame(mask_dir, frame_num):
    fname = f"{frame_num:04d}.pkl.gz"
    path = os.path.join(mask_dir, fname)
    if not os.path.exists(path):
        return {}
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def collect_all_object_ids_from_dir(mask_dir: str) -> set:
    """Scans all per-frame .pkl.gz files in a directory to collect unique object IDs."""
    object_ids = set()
    frame_files = sorted(f for f in os.listdir(mask_dir) if f.endswith(".pkl.gz"))

    for fname in tqdm(frame_files, desc="Collecting object IDs"):
        path = os.path.join(mask_dir, fname)
        with gzip.open(path, "rb") as f:
            frame_data = pickle.load(f)
        object_ids.update(frame_data[list(frame_data.keys())[0]].keys())

    return object_ids


def deterministic_color_from_id(obj_id):
    h = hashlib.md5(str(obj_id).encode()).hexdigest()
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def get_or_assign_color(color_map, obj_id):
    """Assign a deterministic color to the object ID if not already assigned."""
    if obj_id not in color_map:
        color_map[obj_id] = deterministic_color_from_id(obj_id)
    return color_map[obj_id]


def write_video_from_masks_perframe(
    mask_dir,
    output_path,
    image_folder=None,
    fps=30,
    alpha=0.5,
    shape=None,
    use_torch=False,
    device="cuda",
    target_obj_ids=None,
):
    """Writes a video visualizing per-frame RLE mask dicts over optional images."""
    if os.path.exists(output_path):
        print(f"{output_path} exists. Not writing video.")
        return

    overall_start = time.time()

    # Identify available frames
    frame_nums = sorted(
        int(f.split(".")[0]) for f in os.listdir(mask_dir) if f.endswith(".pkl.gz")
    )

    color_map = {}

    if shape is None:
        shape = infer_shape_from_data(None, image_folder)

    preload_start = time.time()
    images = preload_images(image_folder, frame_nums, shape) if image_folder else None
    preload_duration = time.time() - preload_start

    height, width = shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    process_times = []

    for frame_num in tqdm(frame_nums, desc="Writing video"):
        frame_start = time.time()

        frame_masks = load_mask_for_frame(mask_dir, frame_num)
        frame_masks = frame_masks[list(frame_masks.keys())[0]]
        if target_obj_ids is not None:
            frame_masks = {k: v for k, v in frame_masks.items() if k in target_obj_ids}
        # color_map = generate_color_map(sorted(frame_masks.keys()))
        if use_torch:
            rgba = generate_frame_rgba_torch(
                frame_masks, shape, color_map, alpha, device=device
            )
        else:
            rgba = generate_frame_rgba(frame_masks, shape, color_map, alpha)

        if images:
            bg = images[frame_num]
            blended = rgba[..., :3] * rgba[..., 3:4] + bg * (1 - rgba[..., 3:4])
        else:
            blended = rgba[..., :3]

        bgr_out = cv2.cvtColor((blended * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(bgr_out)

        process_times.append(time.time() - frame_start)

    writer.release()
    overall_duration = time.time() - overall_start

    total_frames = len(frame_nums)
    avg_frame_time = np.mean(process_times)
    avg_fps = total_frames / overall_duration

    print(f"✅ Video written to {output_path}")
    print(
        f"📊 Processed {total_frames} frames in {overall_duration:.2f} seconds ({avg_fps:.2f} FPS)"
    )
    print(f"📂 Image preload took {preload_duration:.2f} seconds")
    print(f"🧠 Average processing time per frame: {avg_frame_time:.3f} seconds")


def load_and_merge_video_segments_rle(directory: str) -> Dict[int, Dict[int, Any]]:
    """
    Load and merge video_segments_rle.pkl.gz files from a directory into a single dictionary.

    Each file is expected to contain a nested dictionary:
        {frame_idx: {obj_id: rle_encoded_mask}}

    If multiple files contain the same frame index, their object entries are merged.

    Args:
        directory (str): Directory containing the .pkl.gz files.

    Returns:
        Dict[int, Dict[int, Any]]: Consolidated nested dictionary.
    """
    consolidated = defaultdict(dict)

    files = sorted(
        [f for f in os.listdir(directory) if f.endswith("_video_segments_rle.pkl.gz")]
    )

    for filename in files:
        path = os.path.join(directory, filename)
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)

        for frame_idx, obj_dict in data.items():
            consolidated[frame_idx].update(obj_dict)

    return dict(consolidated)


def main() -> None:
    from directory_dict import (
        directory_dict,
        get_mask_and_frame_and_parent_dirs,
    )

    # vid_key = "vid247_rolling1"
    # mask_dir, frame_dir, parent_output_dir = get_mask_and_frame_and_parent_dirs(
    #     directory_dict, vid_key
    # )
    # # _, summary = read_df_and_summary_objs_from_parent(parent_output_dir)
    # # summary = summary.drop(
    # #     columns=["obj_id_mean", "obj_id_std", "frame_mean", "frame_std", "num_frames"]
    # # )
    # # result_df, model = detect_outliers_isolation_forest(summary_df=summary)

    # write_video_from_masks_perframe(
    #     mask_dir=mask_dir,
    #     output_path=os.path.join(parent_output_dir, "animation_100fps.avi"),
    #     image_folder=frame_dir,
    #     fps=100,
    #     alpha=0.5,
    #     shape=(800, 1280),
    #     use_torch=True,
    #     device="cuda",
    #     target_obj_ids=None,
    # )

    for vid_key in directory_dict.keys():
        try:
            print(f"Handling video key {vid_key}")
            mask_dir, frame_dir, parent_output_dir = get_mask_and_frame_and_parent_dirs(
                directory_dict, vid_key
            )
            write_video_from_masks_perframe(
                mask_dir=mask_dir,
                output_path=os.path.join(
                    parent_output_dir,
                    f"{re.sub('rolling', 'r', vid_key)}_animation_100fps.avi",
                ),
                image_folder=frame_dir,
                fps=100,
                alpha=0.5,
                shape=(800, 1280),
                use_torch=True,
                device="cuda",
                target_obj_ids=None,
            )
        except Exception as e:
            print(e)

    # write_video_from_masks_perframe(
    #     mask_dir=directory,
    #     output_path=os.path.join(parent_output_dir, "inliers_100fps.avi"),
    #     image_folder=image_folder,
    #     fps=100,
    #     alpha=0.5,
    #     shape=(800, 1280),
    #     use_torch=True,
    #     device="cuda",
    #     target_obj_ids=result_df.obj_id.values[
    #         [not i for i in result_df.is_outlier.values]
    #     ],
    # )

    # write_video_from_masks_perframe(
    #     mask_dir=directory,
    #     output_path=os.path.join(parent_output_dir, "outliers_100fps.avi"),
    #     image_folder=image_folder,
    #     fps=100,
    #     alpha=0.5,
    #     shape=(800, 1280),
    #     use_torch=True,
    #     device="cuda",
    #     target_obj_ids=result_df.obj_id.values[result_df.is_outlier.values],
    # )


if __name__ == "__main__":
    main()
