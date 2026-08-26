"""concat_mp3.py - Detect and concatenate MP3 files into a single MP3 audio track.

Scans an input directory recursively for all MP3 files, sorts them,
and combines them into one output MP3 file using ffmpeg.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("concat_mp3")  # type: ignore[assignment]
    if not hasattr(logger, "success"):
        logger.success = logger.info  # type: ignore[attr-defined]


def find_mp3_files(input_dir: Path) -> list[Path]:
    """Recursively discover and sort all MP3 files within a directory.

    Args:
        input_dir: Directory to scan for MP3 files.

    Returns:
        Sorted list of Path objects for all found MP3 files.
    """
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return []

    files = sorted(
        [p for p in input_dir.rglob("*.mp3") if p.is_file()],
        key=lambda p: (str(p.parent).lower(), p.name.lower()),
    )
    logger.info(f"Found {len(files)} MP3 file(s) in '{input_dir}'")
    return files


def probe_duration(file_path: Path) -> float:
    """Probe audio duration in seconds using ffprobe.

    Args:
        file_path: Path to the audio file.

    Returns:
        Duration in seconds as a float, or 0.0 if probing fails.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception as exc:
        logger.warning(f"Could not probe duration for {file_path}: {exc}")
        return 0.0


def format_duration(seconds: float) -> str:
    """Format duration in seconds into human-readable H:M:S format.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string (e.g., '1h 23m 45s' or '3m 12s').
    """
    total_secs = int(seconds)
    hours, remainder = divmod(total_secs, 3600)
    mins, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {mins:02d}m {secs:02d}s"
    return f"{mins}m {secs:02d}s"


def concatenate_mp3_files(
    mp3_files: Sequence[Path],
    output_file: Path,
    bitrate: str = "320k",
) -> Path:
    """Concatenate multiple MP3 files into a single MP3 file via ffmpeg.

    Args:
        mp3_files: Sequence of MP3 file paths in concatenation order.
        output_file: Destination path for the merged MP3 file.
        bitrate: Output audio bitrate (e.g. '320k', '256k', '192k').

    Returns:
        The output Path on success.

    Raises:
        ValueError: If mp3_files list is empty.
        RuntimeError: If ffmpeg concatenation command fails.
    """
    if not mp3_files:
        raise ValueError("No MP3 files provided for concatenation.")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as list_fh:
        list_file_path = Path(list_fh.name)
        for mp3 in mp3_files:
            safe_path = str(mp3.resolve()).replace("\\", "/")
            list_fh.write(f"file '{safe_path}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file_path),
        "-c:a", "libmp3lame",
        "-b:a", bitrate,
        str(output_file),
    ]

    logger.info(f"Concatenating {len(mp3_files)} files into: {output_file}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as err:
        logger.error(f"FFmpeg error: {err.stderr.strip()}")
        raise RuntimeError(f"FFmpeg concatenation failed: {err.stderr.strip()}") from err
    finally:
        list_file_path.unlink(missing_ok=True)

    duration = probe_duration(output_file)
    size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.success(
        f"Concatenation complete: {output_file.name} "
        f"({format_duration(duration)}, {size_mb:.2f} MB)"
    )
    return output_file


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect and concatenate MP3 files in a directory into one MP3 file.",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=Path("archive-02"),
        help="Source directory containing MP3 files (default: archive-02)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("input-mp3"),
        help="Destination directory for concatenated MP3 (default: input-mp3)",
    )
    parser.add_argument(
        "--filename",
        "-f",
        type=str,
        default="concatenated_audio.mp3",
        help="Filename for the merged MP3 file (default: concatenated_audio.mp3)",
    )
    parser.add_argument(
        "--bitrate",
        "-b",
        type=str,
        default="320k",
        help="Audio bitrate for output MP3 (default: 320k)",
    )
    return parser.parse_args()


def main() -> int:
    """Main execution function."""
    args = parse_args()
    input_path = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_path = output_dir / args.filename

    logger.info(f"Scanning '{input_path}' for MP3 files...")
    mp3_files = find_mp3_files(input_path)

    if not mp3_files:
        logger.error(f"No MP3 files found in '{input_path}'. Exiting.")
        return 1

    try:
        concatenate_mp3_files(
            mp3_files=mp3_files,
            output_file=output_path,
            bitrate=args.bitrate,
        )
        return 0
    except Exception as exc:
        logger.exception(f"Failed to concatenate MP3 files: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
