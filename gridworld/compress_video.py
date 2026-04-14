#!/usr/bin/env python3
"""Compress recorded gridworld videos to ~15 seconds for GitHub upload.

Reads the original MP4, samples frames evenly to fit a target duration,
and writes a smaller output file.

Usage:
    # Compress all videos in gridworld/videos/
    python -m gridworld.compress_video

    # Compress a specific file
    python -m gridworld.compress_video --input gridworld/videos/qlearning_phase1_tagger_training.mp4

    # Custom target duration and output dir
    python -m gridworld.compress_video --duration 10 --output-dir gridworld/videos_compressed
"""

import argparse
import glob
import os
import cv2


def compress_video(input_path: str, output_path: str, target_duration: float = 15.0,
                   output_fps: int = 30):
    """Speed up a video to fit within target_duration seconds.

    Args:
        input_path: Path to the original MP4.
        output_path: Where to write the compressed MP4.
        target_duration: Target video length in seconds.
        output_fps: FPS of the output video.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {input_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    if total_frames <= 0:
        print(f"  [SKIP] Empty video: {input_path}")
        cap.release()
        return

    src_duration = total_frames / src_fps
    target_frames = int(target_duration * output_fps)

    # How many source frames to skip per output frame
    step = max(1, total_frames // target_frames)
    actual_out_frames = total_frames // step
    actual_duration = actual_out_frames / output_fps
    speedup = src_duration / actual_duration if actual_duration > 0 else 1

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    frame_idx = 0
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            writer.write(frame)
            written += 1
        frame_idx += 1

    cap.release()
    writer.release()

    in_size = os.path.getsize(input_path) / (1024 * 1024)
    out_size = os.path.getsize(output_path) / (1024 * 1024)

    print(f"  {os.path.basename(input_path)}")
    print(f"    {src_duration:.0f}s -> {actual_duration:.1f}s  "
          f"({speedup:.0f}x speedup)  "
          f"{in_size:.1f}MB -> {out_size:.1f}MB  "
          f"({written} frames)")


def main():
    parser = argparse.ArgumentParser(description="Compress gridworld videos")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Specific video file to compress")
    parser.add_argument("--duration", "-d", type=float, default=15.0,
                        help="Target duration in seconds (default: 15)")
    parser.add_argument("--output-dir", "-o", type=str,
                        default="gridworld/videos_compressed",
                        help="Output directory (default: gridworld/videos_compressed)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output FPS (default: 30)")
    args = parser.parse_args()

    if args.input:
        files = [args.input]
    else:
        files = sorted(glob.glob("gridworld/videos/*.mp4"))

    if not files:
        print("No videos found in gridworld/videos/")
        return

    print(f"Compressing {len(files)} video(s) to ~{args.duration}s each:\n")

    for f in files:
        name = os.path.basename(f)
        out = os.path.join(args.output_dir, name)
        compress_video(f, out, target_duration=args.duration, output_fps=args.fps)
        print()

    print(f"Done. Compressed videos in: {args.output_dir}/")


if __name__ == "__main__":
    main()
