import os

import torch
from sam2.build_sam import (
    build_sam2,
    build_sam2_video_predictor,
)
from sam2.sam2_video_predictor import SAM2VideoPredictor


def initialize_cuda_device():
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print("CUDA device initialized")
    return device


def initialize_sam2_video_predictor(
    video_dir: str,
    sam2_checkpoint: str = "/home/leviathan/cdrucker/ml/sam2/checkpoints/sam2.1_hiera_tiny.pt",
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
    async_loading_frames: bool = True,
    offload_state_to_cpu: bool = True,
    offload_video_to_cpu: bool = True,
) -> tuple[SAM2VideoPredictor, dict]:
    device = initialize_cuda_device()
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(
        video_path=video_dir,
        async_loading_frames=async_loading_frames,
        offload_state_to_cpu=offload_state_to_cpu,
        offload_video_to_cpu=offload_video_to_cpu,
    )
    print("SAM2 video predictor initialized")
    return predictor, inference_state


def scan_in_frame_names(video_dir: str, initial_frame: int, n_frames: int):
    # scan all the JPEG frame names in this directory
    frame_names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    n_frames = min(len(frame_names), n_frames)
    return frame_names[initial_frame : (initial_frame + n_frames)]


def initialize_sam2_mask_generator(
    device,
    sam2_checkpoint: str = "/home/leviathan/cdrucker/ml/sam2/checkpoints/sam2.1_hiera_tiny.pt",
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
):
    # # Implement a regular sam2 model when there is inline mask generation for prompting
    sam2 = build_sam2(
        model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False
    )

    return sam2
