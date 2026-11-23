#!/usr/bin/env python3
"""
Usage: generate_configs.py <template_config.yaml> <video1.avi> [video2.avi ...]

For each video, derives a new config from the template by updating paths.output_dir,
paths.video_dir, and paths.pkl_dir based on the video's number (first 3 characters
of its filename), then writes the new config to its (newly created) output_dir.

Example (brace expansion for videos differing only by number):
    ./generate_configs.py template_config.yaml 364_1p0mlmin_800fps_lambda5_500prep_1000cent.avi
"""

import copy
import os
import sys

import yaml


def basename_no_ext(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    if len(sys.argv) < 3:
        print(
            f"Usage: {sys.argv[0]} <template_config.yaml> <video1.avi> [video2.avi ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    template_path = sys.argv[1]
    video_paths = sys.argv[2:]

    with open(template_path) as f:
        template = yaml.safe_load(f)

    old_video_path = template["paths"]["video_dir"]
    old_video_name = basename_no_ext(old_video_path)
    old_number = old_video_name[:3]

    old_output_dir = template["paths"]["output_dir"]
    old_pkl_dir = template["paths"]["pkl_dir"]

    for video_path in video_paths:
        if not os.path.isfile(video_path):
            print(f"Warning: '{video_path}' not found, skipping.", file=sys.stderr)
            continue

        new_video_path = os.path.abspath(video_path)
        new_video_name = basename_no_ext(new_video_path)
        new_number = new_video_name[:3]

        # Replace the number only in the final path component of output_dir
        parent, base = os.path.split(old_output_dir)
        if old_number not in base:
            print(
                f"Warning: old number '{old_number}' not found in output_dir base '{base}'; "
                f"output_dir left unmodified for '{video_path}'.",
                file=sys.stderr,
            )
            new_base = base
        else:
            new_base = base.replace(old_number, new_number, 1)
        new_output_dir = os.path.join(parent, new_base)

        # Replace the video name substring in pkl_dir
        if old_video_name not in old_pkl_dir:
            print(
                f"Warning: old video name '{old_video_name}' not found in pkl_dir; "
                f"pkl_dir left unmodified for '{video_path}'.",
                file=sys.stderr,
            )
            new_pkl_dir = old_pkl_dir
        else:
            new_pkl_dir = old_pkl_dir.replace(old_video_name, new_video_name)

        new_config = copy.deepcopy(template)
        new_config["paths"]["output_dir"] = new_output_dir
        new_config["paths"]["video_dir"] = new_video_path
        new_config["paths"]["pkl_dir"] = new_pkl_dir

        os.makedirs(new_output_dir, exist_ok=True)
        out_config_path = os.path.join(new_output_dir, "config.yaml")
        with open(out_config_path, "w") as f:
            yaml.dump(new_config, f, sort_keys=False, default_flow_style=False)

        print(f"'{video_path}' -> '{out_config_path}'")


if __name__ == "__main__":
    main()
