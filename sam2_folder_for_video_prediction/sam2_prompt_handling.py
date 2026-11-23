import gzip
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask
from sam2.sam2_video_predictor import SAM2VideoPredictor
from tqdm import tqdm


def apply_crop_to_mask(
    mask: np.ndarray, filename: str, modifications: dict
) -> np.ndarray:
    """Apply cropping (and optional rotation) to a binary mask based on metadata."""
    if filename not in modifications:
        raise ValueError(f"No modification entry found for {filename}")

    mod = modifications[filename]
    y1, y2, x1, x2 = mod["crop_bounds"]
    angle = mod.get("rotation", 0)

    if angle != 0:
        h, w = mask.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        mask = cv2.warpAffine(
            mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
        )

    return mask[y1:y2, x1:x2]


def extract_prompts_from_annotations(
    annotations: List[dict],
    ann_frame_idx: int,
    predictor: SAM2VideoPredictor,
    inference_state: dict,
    n_points_per_mask: int,
    min_area: int,
    max_area: int,
    kernel_size: int,
    is_initial_frame: bool,
    border_proportion: float,
    modifications: Optional[dict] = None,
    add_prompts_to_predictor: bool = True,
) -> Tuple[
    Dict[int, Dict[int, Union[Tuple[np.ndarray, np.ndarray], np.ndarray]]],
    List[int],
    List[torch.Tensor],
]:
    """Extract prompts (point or mask) from annotations and optionally add to predictor."""
    prompts_by_frame = defaultdict(dict)
    out_obj_ids = []
    out_mask_logits = []

    for i, ann in enumerate(annotations):
        if not (min_area <= ann["area"] < max_area):
            continue

        # Decode mask
        m = (
            coco_mask.decode(ann["segmentation"])
            if isinstance(ann["segmentation"], dict)
            else ann["segmentation"]
        )

        # Apply cropping, skip if outside
        if modifications:
            m = apply_crop_to_mask(m, "0000.jpg", modifications)
            if not m.any():
                continue

        h, w = m.shape
        top = m[: int(h * border_proportion), :]
        bottom = m[int(h * (1 - border_proportion)) :, :]
        is_border = np.any(top) or np.any(bottom)

        if not is_initial_frame and not is_border:
            continue

        # Generate deterministic but unique obj_id
        obj_id = int(f"{ann_frame_idx}{i:04}")  # Up to 9999 annotations per frame

        if n_points_per_mask > 0:
            points = sample_points_from_mask(m, n_points_per_mask, kernel_size)
            if points is None:
                continue
            labels = np.ones(n_points_per_mask, dtype=np.int32)
            prompts_by_frame[ann_frame_idx][obj_id] = (points, labels)

            if add_prompts_to_predictor:
                _, new_obj_ids, mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )
        else:
            prompts_by_frame[ann_frame_idx][obj_id] = m
            if add_prompts_to_predictor:
                _, new_obj_ids, mask_logits = predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=obj_id,
                    mask=m,
                )

        if add_prompts_to_predictor:
            out_obj_ids.extend(new_obj_ids)
            out_mask_logits.append(mask_logits)

    return prompts_by_frame, out_obj_ids, out_mask_logits


def sample_points_from_mask(
    mask: np.ndarray, n_points: int, kernel_size: int
) -> Optional[np.ndarray]:
    """Erodes the mask and samples n_points inside the eroded region, or returns None."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    m_eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    pts = np.argwhere(m_eroded > 0)
    if pts.shape[0] < n_points:
        return None
    indices = np.random.choice(pts.shape[0], size=n_points, replace=False)
    return pts[indices][:, [1, 0]]  # Convert (y,x) to (x,y)


def generate_prompts_from_mask_pkl_file(
    predictor: SAM2VideoPredictor,
    inference_state: dict,
    pkl_folder: str,
    ann_frame_idx: int = 0,
    n_points_per_mask: int = 1,
    min_area: int = 0,
    max_area: int = 3000,
    kernel_size: int = 1,
    is_initial_frame: bool = True,
    border_proportion: float = 0.1,
    add_prompts_to_predictor: bool = True,
    modification_json_path: str = None,
) -> Tuple[
    Dict[int, Dict[int, Union[Tuple[np.ndarray, np.ndarray], np.ndarray]]],
    List[int],
    torch.Tensor,
]:
    """Load annotations from pickle and generate prompts for a specific frame."""
    sorted_files = sorted(
        [os.path.join(pkl_folder, name) for name in os.listdir(pkl_folder)]
    )
    ann_frame_idx_str = str(ann_frame_idx).zfill(4) + ".pkl.gz"
    pkl_file = [file for file in sorted_files if ann_frame_idx_str in file][0]
    print(f"Making prompts from file {pkl_file}")

    with gzip.open(pkl_file, "rb") as f:
        annotations = pickle.load(f)

    modifications = None
    if modification_json_path:
        with open(modification_json_path, "r") as json_file:
            modifications = json.load(json_file)

    prompts_by_frame, out_obj_ids, out_mask_logits = extract_prompts_from_annotations(
        annotations=annotations,
        ann_frame_idx=ann_frame_idx,
        predictor=predictor,
        inference_state=inference_state,
        n_points_per_mask=n_points_per_mask,
        min_area=min_area,
        max_area=max_area,
        kernel_size=kernel_size,
        is_initial_frame=is_initial_frame,
        border_proportion=border_proportion,
        modifications=modifications,
        add_prompts_to_predictor=add_prompts_to_predictor,
    )

    if add_prompts_to_predictor:
        mask_tensor = (
            torch.cat(out_mask_logits, dim=0)
            if out_mask_logits
            else torch.empty(
                (0, 1, inference_state["video_height"], inference_state["video_width"])
            )
        )
        return dict(prompts_by_frame), out_obj_ids, mask_tensor

    return dict(prompts_by_frame), [], torch.empty(0)


def collect_prompts_from_masks(
    predictor: SAM2VideoPredictor,
    inference_state: dict,
    pkl_folder: str,
    initial_frame: int = 0,
    max_frame_num_to_track: int = 1926,
    n_frames_for_new_masks: int = 100,
    verbose: bool = True,
    **kwargs,
) -> Dict[int, Dict[int, Union[Tuple[np.ndarray, np.ndarray], np.ndarray]]]:
    """
    Collect prompts from mask pickle files across multiple frames without adding them to the predictor.
    """
    prompts_by_frame = defaultdict(dict)

    ann_frames = range(
        initial_frame,
        min(initial_frame + max_frame_num_to_track, inference_state["num_frames"]),
        n_frames_for_new_masks,
    )
    iterator = tqdm(ann_frames, desc="Collecting prompts") if verbose else ann_frames

    for ann_frame_idx in iterator:
        is_initial_frame = ann_frame_idx == initial_frame

        frame_prompts_dict, _, _ = generate_prompts_from_mask_pkl_file(
            predictor=predictor,
            inference_state=inference_state,
            pkl_folder=pkl_folder,
            ann_frame_idx=ann_frame_idx,
            is_initial_frame=is_initial_frame,
            add_prompts_to_predictor=False,
            **kwargs,
        )

        for frame_idx, obj_prompts in frame_prompts_dict.items():
            prompts_by_frame[frame_idx].update(obj_prompts)

    return dict(prompts_by_frame)


def split_prompts_by_frame(
    prompts_by_frame: Dict[
        int, Dict[int, Union[Tuple[np.ndarray, np.ndarray], np.ndarray]]
    ],
    max_prompts: int,
) -> List[Dict[int, Dict[int, Union[Tuple[np.ndarray, np.ndarray], np.ndarray]]]]:
    """Split a large prompts dictionary into chunks limited by max number of prompts."""
    chunks = []
    current_chunk = defaultdict(dict)
    current_count = 0

    for frame_idx in sorted(prompts_by_frame.keys()):
        frame_prompts = prompts_by_frame[frame_idx]
        frame_count = len(frame_prompts)

        if current_count + frame_count > max_prompts and current_chunk:
            chunks.append(dict(current_chunk))
            current_chunk = defaultdict(dict)
            current_count = 0

        current_chunk[frame_idx] = frame_prompts
        current_count += frame_count

    if current_chunk:
        chunks.append(dict(current_chunk))

    return chunks


def populate_prompts(
    predictor: SAM2VideoPredictor,
    inference_state: dict,
    pkl_folder: str,
    initial_frame: int = 0,
    max_frame_num_to_track: int = 1926,
    n_frames_for_new_masks: int = 100,
    max_n_prompts: int = 125,
    verbose: bool = True,
    **kwargs,
) -> Tuple[
    Dict[int, Dict[int, Union[Tuple[np.ndarray, np.ndarray], np.ndarray]]],
    List[int],
    torch.Tensor,
]:
    """Populate prompts from multiple frames, optionally adding them to the predictor."""
    prompts = defaultdict(dict)
    out_obj_ids = []
    out_mask_logits = torch.empty(
        (0, 1, inference_state["video_height"], inference_state["video_width"])
    )

    ann_frames = range(
        initial_frame, initial_frame + max_frame_num_to_track, n_frames_for_new_masks
    )
    iterator = tqdm(ann_frames, desc="Generating prompts") if verbose else ann_frames

    for ann_frame_idx in iterator:
        is_initial_frame = ann_frame_idx == initial_frame

        frame_prompts_dict, frame_obj_ids, frame_logits = (
            generate_prompts_from_mask_pkl_file(
                predictor=predictor,
                inference_state=inference_state,
                pkl_folder=pkl_folder,
                ann_frame_idx=ann_frame_idx,
                is_initial_frame=is_initial_frame,
                **kwargs,
            )
        )

        for frame_idx, obj_prompts in frame_prompts_dict.items():
            prompts[frame_idx].update(obj_prompts)

        out_obj_ids.extend(frame_obj_ids)
        if frame_logits.numel() > 0:
            out_mask_logits = torch.cat((out_mask_logits, frame_logits), dim=0)

        if sum(len(p) for p in prompts.values()) >= max_n_prompts:
            break

    if verbose:
        print(f"{sum(len(p) for p in prompts.values())} total object prompts generated")

    return dict(prompts), out_obj_ids, out_mask_logits


def apply_prompts_to_predictor(
    predictor: SAM2VideoPredictor,
    inference_state: dict,
    prompts_by_frame: Dict[
        int, Dict[int, Union[Tuple[np.ndarray, np.ndarray], np.ndarray]]
    ],
) -> None:
    """
    Apply a nested dictionary of prompts to the predictor.

    Args:
        predictor: The SAM2 video predictor.
        inference_state: State dict tracking prediction across frames.
        prompts_by_frame: Dict mapping frame_idx to a dict of obj_id: (points, labels) or mask.
    """
    for frame_idx in sorted(prompts_by_frame.keys()):
        for obj_id, prompt in prompts_by_frame[frame_idx].items():
            if isinstance(prompt, tuple):
                points, labels = prompt
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )
            else:
                if len(prompt.shape) > 2:
                    prompt = prompt.squeeze()
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=prompt,
                )


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def filter_redundant_prompts_by_frame(
    prompts: Dict[int, Dict[int, Union[np.ndarray, dict]]],
    video_segments: Dict[int, Dict[int, Union[np.ndarray, dict]]],
    iou_threshold: float = 0.5,
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Filters out prompt masks that are redundant with existing masks in the same frame.

    Args:
        prompts: Nested dict [frame_idx][obj_id] = mask (binary or RLE).
        video_segments: Nested dict [frame_idx][obj_id] = mask (binary or RLE).
        iou_threshold: Threshold above which a prompt is considered redundant.

    Returns:
        A filtered prompts dict with redundant masks removed.
    """
    filtered_prompts = defaultdict(dict)
    total_prompts = 0
    removed_prompts = 0

    for frame_idx, obj_dict in prompts.items():
        existing_masks = video_segments.get(frame_idx, {})
        decoded_existing = [
            coco_mask.decode(m) if isinstance(m, dict) else m
            for m in existing_masks.values()
        ]

        for obj_id, prompt_mask in obj_dict.items():
            total_prompts += 1
            prompt_mask_decoded = (
                coco_mask.decode(prompt_mask)
                if isinstance(prompt_mask, dict)
                else prompt_mask
            )

            is_redundant = False
            for existing_mask in decoded_existing:
                if compute_iou(prompt_mask_decoded, existing_mask) >= iou_threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                filtered_prompts[frame_idx][obj_id] = prompt_mask
            else:
                removed_prompts += 1

    print(
        f"Filtered out {removed_prompts} of {total_prompts} prompts "
        f"({removed_prompts / total_prompts:.1%} redundant)."
    )

    return filtered_prompts
