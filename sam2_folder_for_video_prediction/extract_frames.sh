#!/usr/bin/env bash
set -euo pipefail

# Usage: extract_frames.sh <output_dir> <video1.avi> [video2.avi ...]
#
# Extracts frames from each .avi file into:
#   <output_dir>/<video_basename>/frame_00001.jpg, frame_00002.jpg, ...
#
# Example using brace expansion for videos that differ only by number:
#   ./extract_frames.sh /data/frames run_{01..10}.avi

if [ $# -lt 2 ]; then
    echo "Usage: $0 <output_dir> <video1.avi> [video2.avi ...]" >&2
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg is required but not found in PATH." >&2
    exit 1
fi

output_dir="$1"
shift

mkdir -p "$output_dir"

for video in "$@"; do
    if [ ! -f "$video" ]; then
        echo "Warning: '$video' not found, skipping." >&2
        continue
    fi

    base=$(basename "$video" .avi)
    frame_dir="$output_dir/$base"
    mkdir -p "$frame_dir"

    echo "Extracting frames from '$video' -> '$frame_dir'"
    ffmpeg -i "$video" -q:v 2 -start_number 0 "$frame_dir/%04d.jpg"
done