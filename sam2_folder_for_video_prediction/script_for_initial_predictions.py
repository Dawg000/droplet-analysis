#!/home/leviathan/anaconda3/envs/cd_ml/bin/python
import os
import sys

sys.path.append("/home/leviathan/cdrucker/ml/sam2")
# import runSam2VideoPredictionRemote
# import matplotlib.pyplot as plt
import gzip

# from datetime import datetime
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
from sam2.build_sam import build_sam2  # type: ignore

# from pathlib import Path


def getAviFiles(videoDirectory):
    aviFiles = []
    for filename in os.listdir(videoDirectory):
        if (filename.endswith("avi") or filename.endswith("mp4")) and os.path.isfile(
            os.path.join(videoDirectory, filename)
        ):
            aviFiles.append(os.path.join(videoDirectory, filename))
    return aviFiles


def initializeSAM2MaskGenerator(
    sam2_checkpoint="/home/leviathan/cdrucker/ml/sam2/checkpoints/sam2.1_hiera_tiny.pt",
    model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
    pps=128,
    cnl=2,
    cnpdf=1,
):
    device = torch.device("cuda")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    sam2 = build_sam2(
        model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False
    )
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=pps,
        points_per_batch=800,  # During initial testing, setting to 960 uses like 31GB of memory (out of 32 available)
        pred_iou_thresh=0.01,  # Set to 0.01 to allow for as many identifications as possible
        stability_score_thresh=0.95,
        stability_score_offset=1.0,
        mask_threshold=0.0,
        box_nms_thresh=0.7,
        crop_n_layers=cnl,
        crop_nms_thresh=0.7,
        crop_overlap_ratio=512 / 1500,
        crop_n_points_downscale_factor=cnpdf,
        point_grids=None,
        min_mask_region_area=25,
        output_mode="coco_rle",  # change from binary_mask to coco_rle
        use_m2m=False,
        multimask_output=False,  # If set to false,
    )
    return mask_generator


# Read every Nth frame in an avi file, apply the SAM2 mask generation, then output a
# compressed pkl.gz file with the segmentation results
def runSAM2MaskGenerationOnEveryNthFrameOfAvi(
    mask_generator,
    aviFile: str,
    n: int,
    outputDirectory: str,
    bboxOnly: bool = True,
    objectClassInt: int = 0,
    maxArea: int = 10000,
):
    imageSubdirectory = os.path.join(outputDirectory, Path(aviFile).stem, "images")
    os.makedirs(imageSubdirectory, exist_ok=True)

    annotationSubdirectory = os.path.join(
        outputDirectory, Path(aviFile).stem, "annotationPkls"
    )
    os.makedirs(annotationSubdirectory, exist_ok=True)

    labelSubdirectory = os.path.join(outputDirectory, Path(aviFile).stem, "labels")
    os.makedirs(labelSubdirectory, exist_ok=True)

    # Read in the AVI
    cap = cv2.VideoCapture(aviFile)
    nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Detected {nFrames} frames in {aviFile}")

    # Iterate through every nth frame, save the image, and run SAM2 and save the annotations
    # as a compressed pkl file
    for frame in range(0, nFrames, n):
        imageOutputName = os.path.join(imageSubdirectory, str(frame).zfill(4)) + ".jpg"
        labelOutputName = os.path.join(labelSubdirectory, str(frame).zfill(4)) + ".txt"
        annotationOutputName = (
            os.path.join(annotationSubdirectory, str(frame).zfill(4)) + ".pkl.gz"
        )
        if not (
            os.path.exists(imageOutputName)
            and os.path.exists(labelOutputName)
            and os.path.exists(annotationOutputName)
        ):  # Try to avoid redoing frames that have already been done
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame) - 1)
            _, videoFrame = cap.read()

            cv2.imwrite(imageOutputName, videoFrame)

            # Masks are output in COCO_RLE format to save on memory
            masks = mask_generator.generate(videoFrame)
            h, w = masks[0]["segmentation"]["size"]
            arrayToExport = np.zeros((0, 5), dtype=np.float32)
            for currentDict in masks:
                if currentDict["area"] < maxArea:
                    normalizedBBox = np.array(currentDict["bbox"]) / np.array(
                        [w, h, w, h]
                    )
                    # normalizedBBox[0] = normalizedBBox[0] + normalizedBBox[2] * 0.5  # Shift corner x coord to x_center
                    # normalizedBBox[1] = normalizedBBox[1] + normalizedBBox[3] * 0.5  # Shift corner y coord to y_center
                    labelWithBBox = np.hstack(
                        (np.array([objectClassInt]), normalizedBBox)
                    )
                    arrayToExport = np.vstack((arrayToExport, labelWithBBox))

            np.savetxt(
                labelOutputName,
                arrayToExport,
                fmt=["%.0u", "%.8f", "%.8f", "%.8f", "%.8f"],
                delimiter="\t",
            )

            with gzip.open(annotationOutputName, "wb") as f:
                pickle.dump(masks, f)

            print(f"Completed frame {frame} in video {os.path.basename(aviFile)}")
            del masks
            torch.cuda.empty_cache()
    cap.release()
    # cv2.destroyAllWindows()


def main():
    # videoDirectory = "/home/leviathan/cdrucker/videos"
    # aviFiles = getAviFiles(videoDirectory)
    n = 50

    # # Original with tiny SAM2
    # outputDirectory = (
    #     "/home/leviathan/cdrucker/ml/sam2/outputs/automatic_mask_generator_outputs/20250815"
    # )

    # Testing large SAM2 model
    outputDirectory = "/home/leviathan/cdrucker/ml/sam2/outputs/automatic_mask_generator_outputs/20260714"
    aviFiles = [
        f"/home/leviathan/cdrucker/videos/{i}_0p8mlmin_800fps_lambda1_mu4_500prep_1000cent.avi"
        for i in range(506, 519)
    ]

    os.makedirs(outputDirectory, exist_ok=True)

    # mask_generator = initializeSAM2MaskGenerator(pps=256, cnl=3, cnpdf=2)
    mask_generator = initializeSAM2MaskGenerator(
        sam2_checkpoint="/home/leviathan/cdrucker/ml/sam2/checkpoints/sam2.1_hiera_large.pt",
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        pps=256,
        cnl=2,
        cnpdf=2,
    )

    for aviFile in aviFiles:
        try:
            runSAM2MaskGenerationOnEveryNthFrameOfAvi(
                mask_generator, aviFile, int(n), outputDirectory
            )
        except Exception as e:
            print(f"Failed to run mask generation on {aviFile}\nError: {e}")


if __name__ == "__main__":
    main()
