# run_rolling_sam2
"""
Don't forget that lines 144-145 and 165 have been commented out
in misc.py to try to save on VRAM
"""

import sys

sys.path.append("/home/leviathan/cdrucker/ml/sam2")
import gc
import gzip
import logging
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from pycocotools import mask as mask_utils
from sam2.sam2_video_predictor import SAM2VideoPredictor

# from sam2.utils.amg import rle_to_mask
from sam2_memory_reporting import (
    export_memory_snapshot,
    report_memory_usage,
    start_record_memory_history,
    stop_record_memory_history,
)
from sam2_prompt_handling import apply_prompts_to_predictor, collect_prompts_from_masks
from sam2_video_predictor_init import (
    initialize_sam2_video_predictor,
)
from tqdm import tqdm

# https://pytorch.org/blog/understanding-gpu-memory-1/
logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000


# Type aliases for clarity
FrameIndex = int
ObjectID = int
Mask = np.ndarray
MaskDict = Dict[ObjectID, Mask]
SegmentDict = Dict[FrameIndex, MaskDict]


def get_prompt_slice(
    prompts: SegmentDict, start_frame: int, end_frame: int
) -> SegmentDict:
    """Extract prompts from a subset of frames."""
    return {
        frame: prompts[frame]
        for frame in range(start_frame, end_frame)
        if frame in prompts
    }


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def remove_empty_masks(segments: SegmentDict) -> SegmentDict:
    """Remove object masks that are completely empty (i.e., object left the scene)."""
    cleaned_segments = defaultdict(dict)
    n_empty = 0
    for frame, obj_dict in segments.items():
        for obj_id, mask in obj_dict.items():
            if np.any(mask):
                cleaned_segments[frame][obj_id] = mask
            else:
                n_empty += 1
    print(f"Removed {n_empty} empty masks")
    return dict(cleaned_segments)


def merge_prompts(
    previous_prompts: SegmentDict,
    new_prompts: SegmentDict,
    iou_threshold: float = 0.8,
    max_total_prompts: int = 50,
    deferred_prompts: SegmentDict = None,
    eliminated_objects: List = None,
) -> Tuple[SegmentDict, SegmentDict]:
    """
    Merge previous and new prompts into a single batch of up to `max_total_prompts` unique masks.
    Excess prompts are deferred to be used in subsequent batches.

    Args:
        previous_prompts: Historical prompts from earlier in the video.
        new_prompts: New prompts added at this frame range.
        iou_threshold: IoU threshold to filter out redundant prompts.
        max_total_prompts: Maximum number of object masks allowed in one batch.
        deferred_prompts: Additional prompts carried from previous iterations.

    Returns:
        merged_batch: A SegmentDict of prompts to use in this batch.
        leftovers: Remaining prompts to defer to the next batch.
    """

    print("\n[merge_prompts] Starting merge operation...")
    print(f"- Historical prompts: {sum(len(v) for v in previous_prompts.values())}")
    print(f"- New prompts: {sum(len(v) for v in new_prompts.values())}")
    if deferred_prompts:
        print(f"- Deferred prompts: {sum(len(v) for v in deferred_prompts.values())}")

    all_prompts: SegmentDict = {}
    for d in [previous_prompts, new_prompts, deferred_prompts or {}]:
        for frame, obj_dict in d.items():
            all_prompts.setdefault(frame, {}).update(obj_dict)

    # Flatten all prompts to (frame, obj_id, mask)
    prompt_items = [
        (f, oid, mask) for f, objs in all_prompts.items() for oid, mask in objs.items()
    ]
    print(f"- Total candidate masks before filtering: {len(prompt_items)}")

    # Deduplicate prompts via pairwise IoU
    kept = []
    removed = []
    for i, (f_i, oid_i, mask_i) in enumerate(prompt_items):
        is_duplicate = False
        for f_k, oid_k, mask_k in kept:
            if f_i == f_k:
                iou = compute_iou(mask_i, mask_k)
                if iou >= iou_threshold:
                    is_duplicate = True
                    removed.append((f_i, oid_i))

                    if eliminated_objects is not None:
                        eliminated_objects.append(
                            {
                                "eliminated_id": oid_i,
                                "survived_id": oid_k,
                                "frame": f_i,
                                "iou": iou,
                            }
                        )
                    break
        if not is_duplicate:
            kept.append((f_i, oid_i, mask_i))

    print(f"- Masks after deduplication: {len(kept)}")
    print(f"- Redundant masks removed: {len(removed)}")

    # Construct merged_batch and leftovers
    merged_batch: SegmentDict = defaultdict(dict)
    leftovers: SegmentDict = defaultdict(dict)

    for i, (f, oid, mask) in enumerate(kept):
        if i < max_total_prompts:
            merged_batch[f][oid] = mask
        else:
            leftovers[f][oid] = mask

    print(f"- Merged batch size: {sum(len(v) for v in merged_batch.values())}")
    print(f"- Deferred to next batch: {sum(len(v) for v in leftovers.values())}")

    return dict(merged_batch), dict(leftovers)


def extract_last_frame_masks(segments: SegmentDict, last_frame: int) -> SegmentDict:
    """Extract masks from the final frame of a segment to propagate forward."""
    if last_frame in segments:
        return {last_frame: segments[last_frame]}
    return {}


def format_prompts_for_predictor(
    prompts: SegmentDict,
) -> Tuple[List[int], List[int], List[np.ndarray]]:
    """Convert nested prompt dict into flat format for predictor."""
    frame_idxs = []
    obj_ids = []
    masks = []
    for frame_idx, obj_dict in prompts.items():
        for obj_id, mask in obj_dict.items():
            frame_idxs.append(frame_idx)
            obj_ids.append(obj_id)
            masks.append(mask)
    return frame_idxs, obj_ids, masks


def count_prompts(prompts: SegmentDict) -> int:
    """Count total number of prompt masks."""
    return sum(len(obj_dict) for obj_dict in prompts.values())


def split_prompts_into_batches(
    prompts: SegmentDict, batch_size: int
) -> List[SegmentDict]:
    """Split prompts into batches for separate propagation runs."""
    flat = [
        (frame, obj_id, mask)
        for frame, obj_dict in prompts.items()
        for obj_id, mask in obj_dict.items()
    ]
    batches = []
    for i in range(0, len(flat), batch_size):
        batch = defaultdict(dict)
        for frame, obj_id, mask in flat[i : i + batch_size]:
            batch[frame][obj_id] = mask
        batches.append(dict(batch))
    return batches


def combine_segment_dicts(*dicts: SegmentDict) -> SegmentDict:
    combined = defaultdict(dict)
    for d in dicts:
        for frame, obj_map in d.items():
            combined[frame].update(obj_map)
    return dict(combined)


def iterative_prompt_merging(
    prompt_dict: SegmentDict,
    start_frame: int,
    end_frame: int,
    prev_prompts: SegmentDict,
    max_total_prompts: int,
    iou_threshold: float = 0.8,
    eliminated_objects: List = None,
) -> List[SegmentDict]:
    """
    Iteratively prepare batches of prompts while ensuring each batch doesn't exceed max_total_prompts.
    Historical prompts (from prev_prompts) are prioritized.

    Args:
        prompt_dict: All available prompts for this chunk.
        start_frame: Start frame of this chunk (inclusive).
        end_frame: End frame of this chunk (inclusive).
        prev_prompts: Prompts carried over from the previous chunk.
        max_total_prompts: Max number of objects allowed in each propagation batch.

    Returns:
        A list of SegmentDicts, each representing a batch of prompt masks.
    """

    # Extract only the new prompts relevant for this chunk
    new_prompts = {
        f: objs for f, objs in prompt_dict.items() if start_frame <= f < end_frame
    }

    # Flatten into a list of (frame, obj_id, mask) for deterministic batching
    new_items = [
        (f, oid, m) for f, objs in new_prompts.items() for oid, m in objs.items()
    ]
    deferred_prompts: SegmentDict = {}

    total_new = len(new_items)
    total_prev = sum(len(v) for v in prev_prompts.values())

    print("\n[iterative_prompt_merging] Starting iterative merging...")
    print(f"- Start frame: {start_frame}, End frame: {end_frame}")
    print(f"- Historical prompts available: {total_prev}")
    print(f"- New prompts in range: {total_new}")
    print(f"- Max objects per batch: {max_total_prompts}")

    batches = []
    batch_count = 0
    while new_items or deferred_prompts:
        batch_count += 1
        print(f"\n[Batch {batch_count}]")
        print(f"- Previous prompts: {sum(len(v) for v in prev_prompts.values())}")
        print(f"- New items left: {len(new_items)}")
        print(
            f"- Deferred prompts from previous batch: {sum(len(v) for v in deferred_prompts.values())}"
        )

        # Convert remaining new_items back into SegmentDict
        batch_new_prompts: SegmentDict = {}
        for f, oid, mask in new_items:
            batch_new_prompts.setdefault(f, {})[oid] = mask

        merged_batch, remaining = merge_prompts(
            previous_prompts=prev_prompts,
            new_prompts=batch_new_prompts,
            iou_threshold=iou_threshold,
            max_total_prompts=max_total_prompts,
            deferred_prompts=deferred_prompts,
            eliminated_objects=eliminated_objects,
        )

        merged_count = sum(len(v) for v in merged_batch.values())
        leftover_count = sum(len(v) for v in remaining.values())
        deferred_count = sum(len(v) for v in deferred_prompts.values())

        if merged_count == 0:
            print(
                "⚠️  No valid merged batch could be created (likely over prompt budget)."
            )
            break

        print(f"- Merged batch size: {merged_count}")
        print(f"- Remaining prompts deferred to next batch: {deferred_count}")
        print(f"- Remaining new prompts after merge: {leftover_count}")

        batches.append(merged_batch)

        # Historical prompts are only used in the first batch
        prev_prompts = {}
        # Prepare next iteration
        new_items = [
            (f, oid, m) for f, objs in remaining.items() for oid, m in objs.items()
        ]
        deferred_prompts = {}

    print(f"\n[iterative_prompt_merging] Finished with {len(batches)} total batches.")
    for i, b in enumerate(batches):
        print(f"  • Batch {i + 1}: {sum(len(v) for v in b.values())} objects")

    return batches


def stitch_segment_checkpoints(output_dir: str) -> SegmentDict:
    """Load and merge all segment checkpoint files from a directory."""
    segment_files = sorted(f for f in os.listdir(output_dir) if f.endswith(".pkl.gz"))
    full_segments = defaultdict(dict)
    for fname in tqdm(segment_files, desc="Stitching files"):
        path = os.path.join(output_dir, fname)
        with gzip.open(path, "rb") as f:
            segment = pickle.load(f)
        for frame, obj_dict in segment.items():
            full_segments[frame].update(obj_dict)
    return dict(full_segments)


def get_survival_prompts(
    history_buffer: SegmentDict,
    active_prompts: SegmentDict,
    window: int,
    current_frame: int,
) -> SegmentDict:
    """
    Gather mask prompts from recent history for surviving objects.

    Args:
        history_buffer: Full video_segments up to the current chunk.
        active_prompts: Prompts for surviving objects in the last chunk.
        window: Number of frames to look back.
        current_frame: Frame index at the start of the next chunk.

    Returns:
        SegmentDict for initializing SAM2 before next chunk.
    """
    from collections import defaultdict

    reinit_prompts = defaultdict(dict)
    target_ids = {
        obj_id for frame_dict in active_prompts.values() for obj_id in frame_dict
    }

    for f in range(current_frame - 1, current_frame - window - 1, -1):
        if f in history_buffer:
            for obj_id in target_ids:
                if obj_id in history_buffer[f]:
                    reinit_prompts[f][obj_id] = history_buffer[f][obj_id]

    return dict(reinit_prompts)


def merge_segment_dicts(a: SegmentDict, b: SegmentDict) -> SegmentDict:
    """
    Merge two SegmentDicts (nested dictionaries of {frame: {obj_id: mask}}).
    Entries in `b` take precedence if there is a collision.

    Args:
        a: First SegmentDict.
        b: Second SegmentDict.

    Returns:
        A merged SegmentDict.
    """
    from copy import deepcopy

    merged = deepcopy(a)
    for frame, obj_dict in b.items():
        if frame not in merged:
            merged[frame] = {}
        merged[frame].update(obj_dict)
    return merged


def write_segment_checkpoint(video_segments: SegmentDict, output_dir: str):
    """
    Write a checkpoint of the current video_segments to disk as a gzipped pickle.
    """
    start_frame = min(video_segments.keys())
    end_frame = max(video_segments.keys())
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(
        output_dir, f"frame{start_frame}-{end_frame}_segments.pkl.gz"
    )
    with gzip.open(filename, "wb") as f:
        pickle.dump(video_segments, f)
    print(
        f"[Checkpoint] Saved frames {start_frame}-{end_frame} with {sum(len(v) for v in video_segments.values())} masks."
    )


def save_segments_per_frame_rle(
    video_segments: Dict[int, Dict[int, dict]],
    output_dir: str,
    encode_as_rle: bool = True,
):
    """
    Save each frame's segment masks as a separate .pkl.gz file.

    Args:
        video_segments: Dict[frame_idx][object_id] = RLE_mask
        output_dir: directory to write output files
        encode_as_rle: bool to encode masks as RLE instead of binary masks
    """
    os.makedirs(output_dir, exist_ok=True)

    for frame_idx, obj_masks in tqdm(
        video_segments.items(), desc="Saving segments per frame"
    ):
        out_path = os.path.join(output_dir, f"{frame_idx:04d}.pkl.gz")
        if encode_as_rle:
            obj_masks = {
                k: mask_utils.encode(np.asfortranarray(v.squeeze()))
                for k, v in obj_masks.items()
            }
        with gzip.open(out_path, "wb") as f:
            pickle.dump({frame_idx: obj_masks}, f)


def reconstruct_true_trajectories(
    video_segments: SegmentDict, eliminated_objects: List
) -> SegmentDict:
    """
    Reconstruct true trajectories by merging segments that represent the same physical droplet
    """
    if not eliminated_objects:
        print("No object eliminations found - returning original segments")
        return video_segments

    # Build equivalence graph
    equivalences = {}
    for elimination in eliminated_objects:
        eliminated_id = elimination["eliminated_id"]
        survived_id = elimination["survived_id"]
        equivalences[eliminated_id] = survived_id

    print(f"Built equivalence graph with {len(equivalences)} eliminations")

    # Find all unique object IDs in video_segments
    all_object_ids = set()
    for frame_data in video_segments.values():
        all_object_ids.update(frame_data.keys())

    # Find canonical IDs
    canonical_mapping = {}
    equivalence_classes = {}

    for obj_id in all_object_ids:
        canonical_id = find_earliest_in_chain(obj_id, equivalences)
        canonical_mapping[obj_id] = canonical_id

        if canonical_id not in equivalence_classes:
            equivalence_classes[canonical_id] = []
        equivalence_classes[canonical_id].append(obj_id)

    # Report what we're doing
    merged_count = sum(
        1 for class_members in equivalence_classes.values() if len(class_members) > 1
    )
    print(
        f"Found {len(equivalence_classes)} unique trajectories ({merged_count} will be merged)"
    )

    # Rebuild video_segments with canonical IDs
    reconstructed_segments = defaultdict(dict)
    for frame, obj_dict in video_segments.items():
        for obj_id, mask in obj_dict.items():
            canonical_id = canonical_mapping.get(obj_id, obj_id)

            # Handle potential conflicts (multiple masks mapping to same canonical ID in same frame)
            if canonical_id in reconstructed_segments[frame]:
                print(
                    f"Warning: Frame {frame} has multiple masks for canonical ID {canonical_id}"
                )
                # For now, keep the first one (could implement more sophisticated logic later)
                continue

            reconstructed_segments[frame][canonical_id] = mask

    return dict(reconstructed_segments)


def find_earliest_in_chain(obj_id: int, equivalences: Dict) -> int:
    """Find the earliest (smallest) ID in an equivalence chain"""
    visited = set()
    to_visit = {obj_id}
    equivalence_class = set()

    # Use BFS to find all connected IDs
    while to_visit:
        current = to_visit.pop()
        if current in visited:
            continue
        visited.add(current)
        equivalence_class.add(current)

        # Add forward connections
        if current in equivalences:
            to_visit.add(equivalences[current])

        # Add backward connections
        for elim_id, surv_id in equivalences.items():
            if surv_id == current:
                to_visit.add(elim_id)

    return min(equivalence_class)  # Return earliest (smallest) ID


def run_rolling_propagation(
    predictor: SAM2VideoPredictor,
    inference_state: dict,
    prompt_dict: SegmentDict,
    chunk_size: int = 200,
    max_prompts: int = 50,
    iou_threshold: float = 0.8,
    history_window: int = 0,
    report_memory: bool = False,
    max_frames_for_memory: int = 50,
    output_dir: str = "rolling_checkpoints",
    initial_frame: int = 0,
) -> SegmentDict:
    """
    Run rolling SAM2 propagation in chunks, enforcing max object tracking constraints.

    Args:
        predictor: SAM2VideoPredictor instance.
        inference_state: Inference metadata.
        prompt_dict: Dictionary of prompt masks indexed by frame and object ID.
        chunk_size: Number of frames per propagation window.
        max_prompts: Maximum number of objects tracked during propagation.
        iou_threshold: IoU threshold for merging new prompts.
        history_window: Number of past frames to reuse as prompt masks for surviving objects.
        report_memory: Whether to report memory usage per frame.
        max_frames_for_memory: Memory decay parameter in predictor.

    Returns:
        Complete SegmentDict of all propagated masks.
    """
    from tqdm import tqdm

    total_frames = inference_state["num_frames"]
    video_segments: SegmentDict = {}
    active_prompts: SegmentDict = {}
    eliminated_objects = []

    for chunk_start in tqdm(
        range(initial_frame, total_frames - initial_frame, chunk_size)
    ):
        chunk_end = min(chunk_start + chunk_size, total_frames)
        print(f"\n--- Chunk {chunk_start}–{chunk_end} ---")

        batches = iterative_prompt_merging(
            prompt_dict=prompt_dict,
            start_frame=chunk_start,
            end_frame=chunk_end,
            prev_prompts=active_prompts,
            max_total_prompts=max_prompts,
            iou_threshold=iou_threshold,
            eliminated_objects=eliminated_objects,
        )
        print(f"  → Eliminations so far: {len(eliminated_objects)}")
        print(f"  → Prepared {len(batches)} prompt batches")

        for batch_id, batch in enumerate(batches):
            print(f" Propagation batch {batch_id + 1}/{len(batches)}")

            # Optional: retrieve historical prompts
            if history_window > 0:
                historical_prompts = get_survival_prompts(
                    history_buffer=video_segments,
                    active_prompts=batch,
                    window=history_window,
                    current_frame=chunk_start,
                )
                print(
                    f"  → Loaded {sum(len(v) for v in historical_prompts.values())} reinitialization prompts"
                )
                combined_batch = merge_segment_dicts(historical_prompts, batch)
            else:
                combined_batch = batch

            apply_prompts_to_predictor(
                predictor=predictor,
                inference_state=inference_state,
                prompts_by_frame=combined_batch,
            )

            tprev = -1
            print(f"    [Batch {batch_id + 1}/{len(batches)}]")
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in predictor.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=chunk_start,
                max_frame_num_to_track=chunk_end - chunk_start,
                reverse=False,
                max_frames_for_memory=max_frames_for_memory,
            ):
                # Ensure we merge instead of overwrite
                if out_frame_idx not in video_segments:
                    video_segments[out_frame_idx] = {}
                video_segments[out_frame_idx].update(
                    {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                )

                if report_memory:
                    tprev = report_memory_usage(tprev)

            predictor.reset_state(inference_state)
            torch.cuda.empty_cache()
            gc.collect()

        # Determine next round's active prompts
        last_frame = chunk_end
        active_prompts = extract_last_frame_masks(video_segments, last_frame)
        active_prompts = remove_empty_masks(active_prompts)
        print(
            f"[Chunk {chunk_start}-{chunk_end}] {count_prompts(active_prompts)} active prompt(s) extracted from last frame"
        )

        # write_segment_checkpoint(video_segments, output_dir=output_dir)
        save_segments_per_frame_rle(
            video_segments=video_segments, output_dir=output_dir
        )

        # Retain minimal state to prevent memory overload
        if history_window > 0:
            min_frame = max(0, last_frame - history_window + 1)
            video_segments = {
                f: objs for f, objs in video_segments.items() if f >= min_frame
            }
        elif last_frame >= inference_state["num_frames"] - 1:
            video_segments = video_segments
        else:
            video_segments = {last_frame: video_segments[last_frame]}
        gc.collect()

    print(
        f"Reconstructing trajectories from {len(eliminated_objects)} object eliminations..."
    )
    video_segments = reconstruct_true_trajectories(video_segments, eliminated_objects)

    return video_segments


# region
# def run_rolling_propagation(
#     predictor: SAM2VideoPredictor,
#     inference_state: dict,
#     prompt_dict: SegmentDict,
#     chunk_size: int = 200,
#     max_prompts: int = 50,
#     iou_threshold: float = 0.8,
#     report_memory: bool = False,
#     max_frames_for_memory: int = 50,
#     history_window: int = 5,
#     output_dir: str = "video_checkpoints",
# ) -> None:
#     """
#     Run rolling SAM2 propagation using iterative prompt merging over video chunks.

#     Args:
#         predictor: SAM2VideoPredictor instance.
#         inference_state: SAM2 internal state object.
#         prompt_dict: Dictionary of prompts indexed by frame and object ID.
#         chunk_size: Number of frames per propagation window.
#         max_prompts: Maximum number of prompts allowed per propagation.
#         iou_threshold: Threshold for IoU-based prompt deduplication.
#         report_memory: If True, report GPU memory usage per frame.
#         max_frames_for_memory: How many frames to keep in memory for SAM2.
#         history_window: Number of previous frames to reinitialize surviving object prompts.
#         output_dir: Directory to write video_segments chunk checkpoints.
#     """
#     import os

#     from tqdm import tqdm

#     os.makedirs(output_dir, exist_ok=True)
#     total_frames = inference_state["num_frames"]
#     video_segments: SegmentDict = {}
#     active_prompts: SegmentDict = {}

#     for chunk_start in tqdm(range(0, total_frames, chunk_size), desc="Chunks"):
#         chunk_end = min(chunk_start + chunk_size, total_frames)
#         print(f"\n[Chunk {chunk_start}-{chunk_end}]")

#         # Reinitialize predictor with surviving masks from previous history
#         if active_prompts:
#             survival_prompts = get_survival_prompts(
#                 history_buffer=video_segments,
#                 active_prompts=active_prompts,
#                 window=history_window,
#                 current_frame=chunk_start,
#             )
#             print(
#                 f"  → Loaded {sum(len(v) for v in survival_prompts.values())} reinitialization prompts"
#             )
#             apply_prompts_to_predictor(
#                 predictor=predictor,
#                 inference_state=inference_state,
#                 prompts_by_frame=survival_prompts,
#             )

#         # Prepare prompt batches for this chunk
#         batches = iterative_prompt_merging(
#             prompt_dict,
#             start_frame=chunk_start,
#             end_frame=chunk_end,
#             prev_prompts=active_prompts,
#             max_total_prompts=max_prompts,
#             iou_threshold=iou_threshold,
#         )
#         print(f"  → Prepared {len(batches)} prompt batches")

#         # Process each prompt batch
#         for batch_num, batch in enumerate(batches):
#             tprev = -1
#             print(f"    [Batch {batch_num + 1}/{len(batches)}]")

#             apply_prompts_to_predictor(
#                 predictor=predictor,
#                 inference_state=inference_state,
#                 prompts_by_frame=batch,
#             )

#             for (
#                 out_frame_idx,
#                 out_obj_ids,
#                 out_mask_logits,
#             ) in predictor.propagate_in_video(
#                 inference_state=inference_state,
#                 start_frame_idx=chunk_start,
#                 max_frame_num_to_track=chunk_size,
#                 reverse=False,
#                 max_frames_for_memory=max_frames_for_memory,
#             ):
#                 if out_frame_idx not in video_segments:
#                     video_segments[out_frame_idx] = {}

#                 video_segments[out_frame_idx].update(
#                     {
#                         out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#                         for i, out_obj_id in enumerate(out_obj_ids)
#                     }
#                 )

#                 if report_memory:
#                     tprev = report_memory_usage(tprev)

#             predictor.reset_state(inference_state)
#             torch.cuda.empty_cache()
#             gc.collect()

#         # Prepare prompts for next chunk
#         last_frame = chunk_end
#         active_prompts = extract_last_frame_masks(video_segments, last_frame)
#         active_prompts = remove_empty_masks(active_prompts)
#         total_objects = sum(len(v) for v in active_prompts.values())
#         print(f"  → {total_objects} surviving objects forwarded to next chunk")

#         # Write checkpoint to disk
#         checkpoint_path = os.path.join(
#             output_dir, f"video_segments_{chunk_start}-{chunk_end}.pkl.gz"
#         )
#         with gzip.open(checkpoint_path, "wb") as f:
#             pickle.dump(video_segments, f, protocol=pickle.HIGHEST_PROTOCOL)
#         print(f"  → Checkpoint written to {checkpoint_path}")

#         # # Optionally discard from memory (comment if you want to retain in-memory)
#         # video_segments.clear()


# def run_rolling_propagation(
#     predictor: "SAM2VideoPredictor",
#     inference_state: dict,
#     prompt_dict: SegmentDict,
#     chunk_size: int = 200,
#     max_prompts: int = 50,
#     iou_threshold: float = 0.8,
#     report_memory: bool = False,
#     max_frames_for_memory: int = 50,
#     output_dir: str = "rolling_outputs",
# ) -> None:
#     from tqdm import tqdm

#     os.makedirs(output_dir, exist_ok=True)
#     total_frames = inference_state["num_frames"]
#     active_prompts: SegmentDict = {}

#     for chunk_start in tqdm(range(0, total_frames, chunk_size)):
#         chunk_end = min(chunk_start + chunk_size, total_frames)

#         print(f"[Chunk {chunk_start}-{chunk_end}] Starting iterative merging...")
#         batches = iterative_prompt_merging(
#             prompt_dict=prompt_dict,
#             start_frame=chunk_start,
#             end_frame=chunk_end,
#             prev_prompts=active_prompts,
#             max_total_prompts=max_prompts,
#             iou_threshold=iou_threshold,
#         )
#         print(
#             f"[Chunk {chunk_start}-{chunk_end}] {len(batches)} batch(es) ready for propagation."
#         )

#         video_segments: SegmentDict = defaultdict(dict)

#         for batch_idx, batch in enumerate(batches):
#             print(
#                 f"  [Batch {batch_idx + 1}/{len(batches)}] Preparing {count_prompts(batch)} prompts..."
#             )
#             tprev = -1

#             apply_prompts_to_predictor(
#                 predictor=predictor,
#                 inference_state=inference_state,
#                 prompts_by_frame=batch,
#             )

#             for (
#                 out_frame_idx,
#                 out_obj_ids,
#                 out_mask_logits,
#             ) in predictor.propagate_in_video(
#                 inference_state=inference_state,
#                 start_frame_idx=chunk_start,
#                 max_frame_num_to_track=chunk_size,
#                 reverse=False,
#                 max_frames_for_memory=max_frames_for_memory,
#             ):
#                 frame_dict = {
#                     out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#                     for i, out_obj_id in enumerate(out_obj_ids)
#                 }
#                 if out_frame_idx not in video_segments:
#                     video_segments[out_frame_idx] = {}
#                 video_segments[out_frame_idx].update(frame_dict)

#                 if report_memory:
#                     tprev = report_memory_usage(tprev)

#             predictor.reset_state(inference_state)
#             torch.cuda.empty_cache()
#             gc.collect()

#         checkpoint_path = os.path.join(
#             output_dir, f"segments_{chunk_start:06d}_{chunk_end:06d}.pkl.gz"
#         )
#         with gzip.open(checkpoint_path, "wb") as f:
#             pickle.dump(dict(video_segments), f)
#         print(
#             f"[Chunk {chunk_start}-{chunk_end}] Saved checkpoint to {checkpoint_path}"
#         )

#         last_frame = chunk_end - 1
#         active_prompts = extract_last_frame_masks(video_segments, last_frame)
#         active_prompts = remove_empty_masks(active_prompts)
#         print(
#             f"[Chunk {chunk_start}-{chunk_end}] {count_prompts(active_prompts)} active prompt(s) extracted from last frame"
#         )

#         del video_segments
#         torch.cuda.empty_cache()
#         gc.collect()
#         print(f"[Chunk {chunk_start}-{chunk_end}] Finished. RAM cleared.\n")
# endregion


def main() -> None:
    # video_dir should be a directory of JPEG frames with filenames like <frame_index>.jpg
    # pkl_dir should be a directory of pickled masks stored either as binary masks or as COCO_RLE
    # output_dir should be the desired output directory
    # directories = {
    #     "video_dir": "/home/leviathan/cdrucker/videos/jpegVideoFrames/17758_1_64_0p2mlmin1_lambda5_2krpm_400fps",
    #     "pkl_dir": "/home/leviathan/cdrucker/outputs/sam2OutputsForTraining/20250620_130000/17758_1_64_0p2mlmin1_lambda5_2krpm_400fps/annotationPkls",
    #     "output_dir": "/home/leviathan/cdrucker/ml/sam2/outputs/vid64_rolling4",
    # }

    directories = {
        "video_dir": "/home/leviathan/cdrucker/videos/jpegVideoFrames/17758_1_41_0p2mlmin3_lambda1_largeDrops_400fps",
        "pkl_dir": "/home/leviathan/cdrucker/outputs/sam2OutputsForTraining/20250620_130000/17758_1_41_0p2mlmin3_lambda1_largeDrops_400fps/annotationPkls",
        "output_dir": "/home/leviathan/cdrucker/ml/sam2/outputs/vid41_rolling2",
    }

    # modification_json_path = os.path.join(
    #     directories["video_dir"], "modifications.json"
    # )

    # Number of frames between adding new masks (must be a multiple of 50 currently)
    n_frames_for_new_masks = 50
    initial_frame = 0
    max_frame_num_to_track = 2000

    # Max number of frames to keep in memory bank (older frames are deleted from inference_state)
    max_frames_for_memory = 50

    border_proportion = 0.5
    min_object_area = 25
    max_object_area = 10000
    chunk_size = n_frames_for_new_masks
    max_prompts = 75  # from 100 for 0 history frames
    iou_threshold = 0.8
    history_window = 0

    # tiny SAM2 used by default also
    sam2_checkpoint = (
        "/home/leviathan/cdrucker/ml/sam2/checkpoints/sam2.1_hiera_tiny.pt"
    )
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

    report_memory = True
    # encode_masks_as_rle = True
    # remove_empty_masks = True

    start_record_memory_history()
    (predictor, inference_state) = initialize_sam2_video_predictor(
        video_dir=directories["video_dir"],
        sam2_checkpoint=sam2_checkpoint,
        model_cfg=model_cfg,
        async_loading_frames=True,
        offload_state_to_cpu=True,
        offload_video_to_cpu=True,
    )

    prompts_by_frame = collect_prompts_from_masks(
        predictor=predictor,
        inference_state=inference_state,
        pkl_folder=directories["pkl_dir"],
        initial_frame=initial_frame,
        max_frame_num_to_track=max_frame_num_to_track,
        n_frames_for_new_masks=n_frames_for_new_masks,
        verbose=True,
        min_area=min_object_area,
        max_area=max_object_area,
        border_proportion=border_proportion,
        # modification_json_path=modification_json_path,
        n_points_per_mask=0,
    )

    video_segments = run_rolling_propagation(
        predictor=predictor,
        inference_state=inference_state,
        prompt_dict=prompts_by_frame,
        chunk_size=chunk_size,
        max_prompts=max_prompts,
        iou_threshold=iou_threshold,
        report_memory=report_memory,
        max_frames_for_memory=max_frames_for_memory,
        history_window=history_window,
        output_dir=os.path.join(directories["output_dir"], "per_frame_video_segments"),
        initial_frame=initial_frame,
    )

    # Reset state and free memory
    predictor.reset_state(inference_state)
    del video_segments
    torch.cuda.empty_cache()
    gc.collect()

    export_memory_snapshot()
    stop_record_memory_history()


if __name__ == "__main__":
    main()
