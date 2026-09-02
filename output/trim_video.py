"""Cap videos in this folder at 11:59:00 without re-encoding them.

Run ``python trim_video.py`` from this folder to process every supported video,
or pass one or more video paths. A temporary stream-copy file is required while
FFmpeg rebuilds the container; it replaces the original only after validation.
No permanent duplicate is kept.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


MAX_DURATION_SECONDS = (11 * 60 * 60) + (59 * 60)
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
TEMP_SPACE_MARGIN_BYTES = 1024 * 1024 * 1024


def _find_tool(name: str) -> str:
    """Return an FFmpeg executable path or raise a clear error."""
    tool = shutil.which(name)
    if tool is None:
        raise RuntimeError(f"{name} was not found on PATH. Install FFmpeg or add it to PATH.")
    return tool


def get_duration_seconds(video_path: Path, ffprobe: str) -> float:
    """Return the duration of ``video_path`` in seconds."""
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return float(json.loads(completed.stdout)["format"]["duration"])


def ensure_temporary_space(video_path: Path, duration: float) -> None:
    """Raise an error before trimming when the video directory lacks working space."""
    estimated_output_bytes = int(video_path.stat().st_size * MAX_DURATION_SECONDS / duration)
    required_bytes = estimated_output_bytes + TEMP_SPACE_MARGIN_BYTES
    free_bytes = shutil.disk_usage(video_path.parent).free
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Not enough free space to safely trim {video_path.name}: "
            f"need about {required_bytes / 1_000_000_000:.1f} GB, but only "
            f"{free_bytes / 1_000_000_000:.1f} GB is available."
        )


def trim_video(video_path: Path, ffmpeg: str, ffprobe: str) -> bool:
    """Trim an over-limit video, replacing it only after successful validation.

    Returns:
        True when the original file was replaced; otherwise False.
    """
    duration = get_duration_seconds(video_path, ffprobe)
    if duration <= MAX_DURATION_SECONDS:
        print(f"Keeping {video_path.name}: {duration:.3f}s is within the limit.")
        return False

    ensure_temporary_space(video_path, duration)
    with tempfile.NamedTemporaryFile(
        dir=video_path.parent,
        prefix=f".{video_path.stem}.trimming-",
        suffix=video_path.suffix,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        print(f"Trimming {video_path.name}: {duration:.3f}s -> {MAX_DURATION_SECONDS}s.")
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(video_path),
                "-map",
                "0",
                "-t",
                str(MAX_DURATION_SECONDS),
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                str(temporary_path),
            ],
            check=True,
        )
        trimmed_duration = get_duration_seconds(temporary_path, ffprobe)
        if trimmed_duration > MAX_DURATION_SECONDS + 0.1:
            raise RuntimeError(
                f"Refusing to replace {video_path.name}: trimmed duration "
                f"({trimmed_duration:.3f}s) exceeds the limit."
            )

        os.replace(temporary_path, video_path)
        print(f"Replaced {video_path.name} with the 11:59:00-capped version.")
        return True
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse optional video paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "videos",
        nargs="*",
        type=Path,
        help="Videos to process. Defaults to supported videos beside this script.",
    )
    return parser.parse_args()


def main() -> int:
    """Process selected videos and return a shell-compatible status code."""
    args = parse_args()
    script_directory = Path(__file__).resolve().parent
    videos = args.videos or sorted(
        path for path in script_directory.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        print("No supported videos found.")
        return 0

    try:
        ffmpeg = _find_tool("ffmpeg")
        ffprobe = _find_tool("ffprobe")
        for video_path in videos:
            trim_video(video_path.resolve(), ffmpeg, ffprobe)
    except (RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
