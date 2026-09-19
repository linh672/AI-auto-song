# -*- coding: utf-8 -*-
"""batch_video.py - Batch-process all input videos with audio from gradio_outputs.

Loops through every MP4 in ``input/``, processing each with exactly
BATCH_SIZE audio files from ``gradio_outputs/``.  Each batch renders
a final video directly into a dedicated archive subfolder and safely
moves used source files after successful completion.

Usage::

    python batch_video.py
    python batch_video.py --batch-size 50
    python batch_video.py --off-step-3

Archive structure per batch::

    archive/<video_stem>_batch_<timestamp>/
        <timestamp>_final.mp4        # rendered output
        <video_name>.mp4             # used input video
        used_audio/                  # 100 consumed audio files
            batch_001/track_01.mp3
            ...
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from make_video import (
    ARCHIVE_DIR,
    GRADIO_OUTPUTS_DIR,
    INPUT_DIR,
    _collect_mp3s,
    _format_seconds,
    _safe_move_to_archive,
    main,
)

BATCH_SIZE = 100


def _list_input_videos(input_dir: Path) -> list[Path]:
    """Return all MP4 files in *input_dir*, sorted alphabetically.

    Args:
        input_dir: Directory containing input video files.

    Returns:
        Sorted list of MP4 paths (may be empty).
    """
    return sorted(input_dir.glob("*.mp4"))


def _take_batch(pool: list[Path], size: int) -> list[Path]:
    """Remove and return the first *size* items from *pool* in-place.

    Args:
        pool: Mutable list of available audio file paths.
        size: Number of items to take.

    Returns:
        List of exactly *size* paths taken from the front of *pool*.
    """
    batch = pool[:size]
    del pool[:size]
    return batch


def _archive_batch_files(
    input_video: Path,
    used_audio: list[Path],
    batch_dir: Path,
) -> None:
    """Move the used input video and audio files into the batch archive folder.

    Args:
        input_video: Source video file to archive.
        used_audio: Audio files consumed by this batch.
        batch_dir: Destination archive directory for this batch.
    """
    # Move input video into batch archive root
    _safe_move_to_archive(input_video, batch_dir)
    print(f"      Archived video  : {input_video.name}")

    # Move audio files into used_audio/ subfolder
    audio_dst = batch_dir / "used_audio"
    audio_dst.mkdir(parents=True, exist_ok=True)
    for audio_file in used_audio:
        _safe_move_to_archive(audio_file, audio_dst)

    # Clean up empty parent directories left behind in gradio_outputs
    for audio_file in used_audio:
        parent = audio_file.parent
        try:
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass

    print(f"      Archived audio  : {len(used_audio)} files -> used_audio/")


def _parse_batch_args() -> argparse.Namespace:
    """Parse command-line options for the batch video workflow."""
    parser = argparse.ArgumentParser(
        description="Batch-process all input videos with audio from gradio_outputs."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of audio files per video (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--off-step-3",
        action="store_true",
        help="Skip step 3: visible-mark and SynthID cleanup.",
    )
    return parser.parse_args()


def batch_main(
    *,
    batch_size: int = BATCH_SIZE,
    step_3_enabled: bool = True,
    input_dir: Path = INPUT_DIR,
    gradio_outputs_dir: Path = GRADIO_OUTPUTS_DIR,
    archive_dir: Path = ARCHIVE_DIR,
) -> None:
    """Process all input videos in batches of *batch_size* audio files each.

    Args:
        batch_size: Number of audio files to use per video.
        step_3_enabled: Whether to run watermark cleanup in step 3.
        input_dir: Directory containing input MP4 videos.
        gradio_outputs_dir: Directory containing generated audio files.
        archive_dir: Root archive directory for completed batches.
    """
    start = time.perf_counter()
    print("=" * 60)
    print("  ACE-Step - Batch Video Builder")
    print("=" * 60)

    videos = _list_input_videos(input_dir)
    if not videos:
        print("\n[INFO] No input videos found. Nothing to do.")
        return

    # Collect full audio pool once; we slice from it per batch
    try:
        audio_pool = _collect_mp3s(gradio_outputs_dir)
    except FileNotFoundError:
        print("\n[INFO] No audio files found in gradio_outputs/. Nothing to do.")
        return

    print(f"\n  Input videos : {len(videos)}")
    print(f"  Audio files  : {len(audio_pool)}")
    print(f"  Batch size   : {batch_size}")
    max_batches = len(audio_pool) // batch_size
    print(f"  Max batches  : {max_batches} (with {len(audio_pool) % batch_size} leftover)")
    print()

    completed = 0
    for idx, video in enumerate(videos, 1):
        if len(audio_pool) < batch_size:
            print(
                f"[WARN] Only {len(audio_pool)} audio file(s) remain "
                f"(need {batch_size}). Stopping batch loop."
            )
            break

        batch_audio = _take_batch(audio_pool, batch_size)
        timestamp = int(time.time())
        batch_name = f"{video.stem}_batch_{timestamp}"
        batch_dir = archive_dir / batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)

        print("-" * 60)
        print(f"  BATCH {idx}/{len(videos)}: {video.name}")
        print(f"  Audio files  : {batch_size} (pool remaining: {len(audio_pool)})")
        print(f"  Archive      : {batch_dir.name}/")
        print("-" * 60)

        try:
            main(
                step_3_enabled=step_3_enabled,
                archive_enabled=False,
                input_video=video,
                mp3_files=batch_audio,
                output_dir=batch_dir,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"\n[ERROR] Batch {idx} failed: {exc}", file=sys.stderr)
            print("        Skipping archive step — source files left in place.")
            # Return unused audio to the pool so they aren't lost
            audio_pool = batch_audio + audio_pool
            continue

        # Archive only after successful render
        try:
            _archive_batch_files(video, batch_audio, batch_dir)
        except OSError as exc:
            print(f"\n[WARN] Archive step had errors: {exc}", file=sys.stderr)

        completed += 1
        print()

    elapsed = time.perf_counter() - start
    print("=" * 60)
    print(f"  Batch run complete: {completed}/{len(videos)} videos processed")
    print(f"  Total time: {_format_seconds(elapsed)}")
    print(f"  Audio files remaining in gradio_outputs/: {len(audio_pool)}")
    print("=" * 60)


if __name__ == "__main__":
    args = _parse_batch_args()
    batch_main(
        batch_size=args.batch_size,
        step_3_enabled=not args.off_step_3,
    )
