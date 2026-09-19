# -*- coding: utf-8 -*-
"""batch_video.py - Batch-process all input videos with audio from gradio_outputs.

Loops through every MP4 in ``input/``, processing each with exactly
BATCH_SIZE audio files from ``gradio_outputs/``. Each batch renders
a final video directly into a dedicated archive subfolder and moves
used source files and folders after successful completion.
"""
from __future__ import annotations

import argparse
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
    """Return all MP4 files in *input_dir*, sorted alphabetically."""
    return sorted(input_dir.glob("*.mp4"))


def _take_batch(pool: list[Path], size: int) -> list[Path]:
    """Remove and return the first *size* items from *pool* in-place."""
    batch = pool[:size]
    del pool[:size]
    return batch


def _archive_batch_files(
    input_video: Path,
    used_audio: list[Path],
    batch_dir: Path,
    gradio_outputs_dir: Path = GRADIO_OUTPUTS_DIR,
) -> None:
    """Move used input video and audio folders into the batch archive.

    Args:
        input_video: Source video file to archive.
        used_audio: Audio files consumed by this batch.
        batch_dir: Destination archive directory for this batch.
        gradio_outputs_dir: Directory containing generated audio files.
    """
    _safe_move_to_archive(input_video, batch_dir)
    print(f"      Archived video  : {input_video.name}")

    audio_dst = batch_dir / "used_audio"
    audio_dst.mkdir(parents=True, exist_ok=True)

    used_set = {f.resolve() for f in used_audio}
    moved_items: set[Path] = set()

    for audio_file in used_audio:
        try:
            rel = audio_file.resolve().relative_to(gradio_outputs_dir.resolve())
            top_item = gradio_outputs_dir / rel.parts[0]
        except (ValueError, IndexError):
            top_item = audio_file.parent if audio_file.parent != gradio_outputs_dir else audio_file

        if top_item == gradio_outputs_dir or not top_item.exists() or top_item in moved_items:
            continue

        if top_item.is_dir():
            remaining_mp3s = {p.resolve() for p in top_item.rglob("*.mp3")} - used_set
            if not remaining_mp3s:
                _safe_move_to_archive(top_item, audio_dst)
                moved_items.add(top_item)
            else:
                target_sub = audio_dst / top_item.name
                target_sub.mkdir(parents=True, exist_ok=True)
                for sidecar in top_item.glob(f"{audio_file.stem}.*"):
                    _safe_move_to_archive(sidecar, target_sub)
                moved_items.add(audio_file)
        else:
            _safe_move_to_archive(top_item, audio_dst)
            moved_items.add(top_item)

    for audio_file in used_audio:
        parent = audio_file.parent
        try:
            if parent.exists() and parent != gradio_outputs_dir and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass

    print(f"      Archived audio  : {len(moved_items)} folders/files -> used_audio/")


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
    """Process all input videos in batches of *batch_size* audio files each."""
    start = time.perf_counter()
    print("=" * 60)
    print("  ACE-Step - Batch Video Builder")
    print("=" * 60)

    videos = _list_input_videos(input_dir)
    if not videos:
        print("\n[INFO] No input videos found. Nothing to do.")
        return

    try:
        audio_pool = _collect_mp3s(gradio_outputs_dir)
    except FileNotFoundError:
        print("\n[INFO] No audio files found in gradio_outputs/. Nothing to do.")
        return

    print(f"\n  Input videos : {len(videos)}")
    print(f"  Audio files  : {len(audio_pool)}")
    print(f"  Batch size   : {batch_size}")
    max_batches = len(audio_pool) // batch_size
    print(f"  Max batches  : {max_batches} (with {len(audio_pool) % batch_size} leftover)\n")

    completed = 0
    for idx, video in enumerate(videos, 1):
        if len(audio_pool) < batch_size:
            print(f"[WARN] Only {len(audio_pool)} audio file(s) remain (need {batch_size}). Stopping batch loop.")
            break

        batch_audio = _take_batch(audio_pool, batch_size)
        timestamp = int(time.time())
        batch_dir = archive_dir / f"{video.stem}_batch_{timestamp}"
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
            audio_pool = batch_audio + audio_pool
            continue

        try:
            _archive_batch_files(video, batch_audio, batch_dir, gradio_outputs_dir=gradio_outputs_dir)
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
    batch_main(batch_size=args.batch_size, step_3_enabled=not args.off_step_3)
