# -*- coding: utf-8 -*-
"""make_video.py - Draft all MP3s from gradio_outputs and build a video.

Workflow
--------
1. Scan ``gradio_outputs/`` recursively for every ``*.mp3`` file.
2. Probe all MP3 durations and input video resolution in parallel.
3. Strip container-level AI metadata (C2PA, XMP, EXIF, mov provenance atoms).
4. Mitigate deep AI watermarks (FAST: FFT notch / PARANOID: SDXL-Turbo diffusion).
5. Upscale the input video in ``input/`` to 1080p if below 1080p (closed-GOP, no B-frames).
6. Concatenate MP3s (in folder-name order) into a single AAC audio track.
7. Loop-extend the 1080p video via MPEG-TS stream-copy to match audio duration.
8. Merge the looped video with the concatenated audio (zero re-encode) and write
   the result to ``output/<timestamp>_final.mp4``.

Hardware targets
----------------
- CPU : Intel i9-14HX  (24 cores / 32 threads)  <- primary encoder
- GPU : RTX 4060 8 GB  (NVENC requires driver >= 570.0 for ffmpeg API 13.0)
- RAM : 32 GB

Encoder selection (auto-detected at runtime)
--------------------------------------------
1. h264_nvenc  -- RTX 4060 NVENC (requires NVIDIA driver >= 570.0)
2. libx264     -- CPU fallback, preset=slow for short clip, ultrafast muxing

Optimisations applied
----------------------
- Parallel ffprobe calls (ThreadPoolExecutor, up to CPU_THREADS workers).
- Lossless container metadata strip (C2PA/XMP/EXIF zero re-encode).
- CPU-parallel FFT notch filter (FAST) or VRAM-safe SDXL-Turbo diffusion pass (PARANOID).
- Resolution pre-check before AI upscale: skips Real-ESRGAN frame extraction when source is already >= 1080p.
- Upscale the short input video ONCE before looping using NVENC/libx264 with closed GOP and no B-frames.
- Stream-copy looped video (-c:v copy with -stream_loop -1 via MPEG-TS) to mux 4+ hours in seconds without re-encoding.
- Audio fast path: when all MP3s share the same codec/sample-rate/channels, they are concatenated via
  a single lossless stream-copy pass (~2s instead of ~11 minutes of AAC transcoding).
- Fallback path: parallel AAC transcode across CPU_THREADS workers, then lossless chunk concat.
- Bounded stderr ring buffer in progress monitoring to prevent memory leaks during long renders.
- NVENC pre-check via ``nvidia-smi`` driver version before attempting upscale encode.
- ``-threads`` tuned for i9-14HX P-core count.

Dependencies
------------
Standard library, **ffmpeg >= 6.0**, **numpy**, **scipy**, and optional **diffusers** (for PARANOID mode).
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from watermark_clean import deep_clean_frames, strip_container_metadata

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.resolve()
GRADIO_OUTPUTS_DIR = ROOT_DIR / "gradio_outputs"
INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output"

# i9-14HX: 8 P-cores + 16 E-cores = 24 cores / 32 threads.
# ffmpeg benefits most from P-core count for decode; keep 16 for encode headroom.
CPU_THREADS = 16

# NVENC requires NVIDIA driver >= 570.0 (ffmpeg NVENC API 13.0).
_NVENC_MIN_DRIVER = (570, 0)

# NVENC preset: p1 = fastest (lowest quality), p7 = slowest (best quality).
# p2 gives great speed with acceptable quality for a music video.
NVENC_PRESET = "p2"
NVENC_TUNE = "hq"       # high-quality tuning mode
NVENC_CQ = "23"         # constant quality (0=best, 51=worst; ~18-28 is typical)

# AAC audio bitrate for concatenated track (320k for pristine music quality)
AUDIO_BITRATE = "320k"

# Watermark Mitigation Mode
# 'fast'    -> CPU 2D-FFT Butterworth notch (~18ms/frame)
# 'paranoid'-> GPU SDXL-Turbo + ControlNet Canny diffusion pass (~3.2s/frame)
# 'off'     -> Skip frame-level cleaning (metadata strip only)
WATERMARK_CLEAN_MODE = os.environ.get("ACE_STEP_CLEAN_MODE", "fast")

# AI enhancement is enabled automatically when the bundled Real-ESRGAN executable exists.
# Set ACE_STEP_AI_UPSCALE=0 to use the faster Lanczos-only fallback.
REAL_ESRGAN_EXECUTABLE = ROOT_DIR / "tools" / "realesrgan-ncnn-vulkan.exe"
AI_UPSCALE_ENABLED = os.environ.get("ACE_STEP_AI_UPSCALE", "1") != "0"
AI_UPSCALE_SCALE = "2"

# ---------------------------------------------------------------------------
# GPU encoder detection
# ---------------------------------------------------------------------------


def _nvidia_driver_version() -> tuple[int, int] | None:
    """Return the installed NVIDIA driver version as (major, minor), or None.

    Uses ``nvidia-smi`` which is always available when an NVIDIA driver is
    installed on Windows.

    Returns:
        Tuple like ``(570, 86)`` or ``None`` if nvidia-smi is not found.
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    raw = result.stdout.strip().split(".")[:2]
    try:
        return (int(raw[0]), int(raw[1]))
    except (ValueError, IndexError):
        return None


def _detect_gpu_encoder() -> str | None:
    """Return ``'h264_nvenc'`` if NVENC is supported by the installed driver.

    Performs a driver version pre-check so the script never tries NVENC when
    the driver is too old (avoids a noisy ffmpeg error and wasted time).

    Returns:
        ``'h264_nvenc'`` or ``None``.
    """
    driver = _nvidia_driver_version()
    if driver is None:
        return None
    if driver < _NVENC_MIN_DRIVER:
        return None
    # Verify the encoder is also compiled into this ffmpeg build
    result = subprocess.run(
        ["ffmpeg", "-encoders", "-v", "quiet"],
        capture_output=True,
        text=True,
    )
    return "h264_nvenc" if "h264_nvenc" in result.stdout else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_ffmpeg() -> None:
    """Raise RuntimeError if ffmpeg/ffprobe is not available on PATH."""
    for tool in ("ffmpeg", "ffprobe"):
        result = subprocess.run(
            [tool, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{tool} not found on PATH.\n"
                "  Windows: https://www.gyan.dev/ffmpeg/builds/\n"
                "  Add the bin/ folder to your system PATH and restart."
            )


def _collect_mp3s(root: Path) -> list[Path]:
    """Return all *.mp3 files under *root*, sorted by parent-folder name then filename.

    Args:
        root: Root directory to search.

    Returns:
        Sorted list of MP3 paths.

    Raises:
        FileNotFoundError: If no MP3 files exist under *root*.
    """
    pattern = str(root / "**" / "*.mp3")
    files = [Path(p) for p in glob.glob(pattern, recursive=True)]
    files.sort(key=lambda p: (p.parent.name, p.name))
    if not files:
        raise FileNotFoundError(f"No MP3 files found under: {root}")
    return files


def _probe_duration(path: Path) -> float:
    """Return the duration of one audio/video file in seconds via ffprobe.

    Args:
        path: File to probe.

    Returns:
        Duration in seconds as a float.

    Raises:
        RuntimeError: If ffprobe exits with a non-zero code.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}:\n{result.stderr.strip()}")
    return float(result.stdout.strip())


def _probe_durations_parallel(paths: list[Path]) -> dict[Path, float]:
    """Probe durations of multiple files in parallel using ThreadPoolExecutor.

    Launches up to ``min(len(paths), CPU_THREADS)`` concurrent ffprobe processes,
    which fully saturates the i9-14HX's E-cores on I/O-bound probe work.

    Args:
        paths: List of file paths to probe.

    Returns:
        Mapping of ``{path: duration_seconds}``.
    """
    workers = min(len(paths), CPU_THREADS)
    results: dict[Path, float] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_path = {pool.submit(_probe_duration, p): p for p in paths}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            results[path] = future.result()
    return results


def _format_seconds(seconds: float) -> str:
    """Format duration in seconds into a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like '1h 23m 45s' or '3m 12s'.
    """
    total_secs = int(seconds)
    hours, remainder = divmod(total_secs, 3600)
    mins, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {mins:02d}m {secs:02d}s"
    return f"{mins}m {secs:02d}s"


def _probe_audio_properties(path: Path) -> tuple[str, str, str]:
    """Return (codec_name, sample_rate, channels) for primary audio stream.

    Args:
        path: Path to audio file.

    Returns:
        Tuple of (codec_name, sample_rate, channels).
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels",
        "-of", "csv=s=x:p=0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return ("", "", "")
    parts = result.stdout.strip().split("x")
    if len(parts) == 3:
        return (parts[0], parts[1], parts[2])
    return ("", "", "")


def _transcode_audio_chunk(src: Path, dst: Path) -> None:
    """Transcode a single audio file to AAC chunk in a worker thread.

    Args:
        src: Input audio file path.
        dst: Output AAC audio file path.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        "-i", str(src),
        "-ar", "44100",
        "-ac", "2",
        "-c:a", "aac",
        "-b:a", AUDIO_BITRATE,
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to transcode {src.name}:\n{result.stderr.strip()}")


def _concat_audio(
    mp3_files: list[Path],
    out_dir: Path,
    expected_duration: float | None = None,
) -> tuple[float, Path, str]:
    """Concatenate *mp3_files* into a single audio file with maximum efficiency.

    If all files share the same audio codec, sample rate, and channels, performs
    an instantaneous lossless stream copy (takes ~1-2s). If files have mixed formats,
    transcodes them concurrently across CPU threads before concatenating.

    Args:
        mp3_files: Ordered list of audio paths to concatenate.
        out_dir: Directory for temporary and output audio files.
        expected_duration: Optional total expected duration in seconds.

    Returns:
        Tuple of (total_duration_seconds, output_audio_path, codec_description).
    """
    # Probe audio properties of first file and check consistency
    first_props = _probe_audio_properties(mp3_files[0])
    all_same = True
    if len(mp3_files) > 1:
        with ThreadPoolExecutor(max_workers=min(len(mp3_files), CPU_THREADS)) as pool:
            props_list = list(pool.map(_probe_audio_properties, mp3_files))
        all_same = all(p == first_props and p[0] != "" for p in props_list)

    if all_same and first_props[0] in {"mp3", "aac", "m4a"}:
        # Fast path: 100% Lossless stream-copy concat (zero re-encode, ~1-2s)
        codec = first_props[0]
        ext = "mp3" if codec == "mp3" else "m4a"
        out_path = out_dir / f"concat_audio.{ext}"
        list_file = out_dir / "audio_list.txt"
        with list_file.open("w", encoding="utf-8") as fh:
            for f in mp3_files:
                safe = str(f.resolve()).replace("\\", "/")
                fh.write(f"file '{safe}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(out_path),
        ]
        _run_ffmpeg(cmd, "Lossless audio stream concat")
        list_file.unlink(missing_ok=True)
        codec_desc = f"{codec.upper()} (lossless copy, zero re-encode)"
    else:
        # Parallel transcode path: transcode chunks across all CPU cores
        print(f"      Parallel transcoding {len(mp3_files)} chunks across {CPU_THREADS} workers...")
        chunks_dir = out_dir / "audio_chunks"
        chunks_dir.mkdir(exist_ok=True)
        chunk_paths: list[Path] = []

        tasks = []
        for i, src in enumerate(mp3_files):
            dst = chunks_dir / f"chunk_{i:05d}.m4a"
            chunk_paths.append(dst)
            tasks.append((src, dst))

        with ThreadPoolExecutor(max_workers=CPU_THREADS) as pool:
            futures = [pool.submit(_transcode_audio_chunk, s, d) for s, d in tasks]
            for future in as_completed(futures):
                future.result()

        out_path = out_dir / "concat_audio.m4a"
        list_file = out_dir / "audio_chunks_list.txt"
        with list_file.open("w", encoding="utf-8") as fh:
            for chunk in chunk_paths:
                safe = str(chunk.resolve()).replace("\\", "/")
                fh.write(f"file '{safe}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(out_path),
        ]
        _run_ffmpeg(cmd, "Concatenating AAC chunks")
        list_file.unlink(missing_ok=True)
        codec_desc = f"AAC {AUDIO_BITRATE}"

    duration = _probe_duration(out_path)
    print(f"      Total audio duration: {_format_seconds(duration)}  ({duration:.2f}s)")
    return duration, out_path, codec_desc


def _probe_resolution(path: Path) -> tuple[int, int]:
    """Return the (width, height) of the primary video stream in *path*.

    Args:
        path: Path to video file.

    Returns:
        Tuple of (width, height).

    Raises:
        RuntimeError: If ffprobe fails or output cannot be parsed.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe resolution failed for {path}:\n{result.stderr.strip()}")
    try:
        parts = result.stdout.strip().split("x")
        return int(parts[0]), int(parts[1])
    except Exception as exc:
        raise RuntimeError(f"Could not parse resolution from {path}: {result.stdout.strip()}") from exc


def _build_cover_scale_filter(target_w: int, target_h: int) -> str:
    """Return an ffmpeg filter that fills a frame with a centered crop.

    Args:
        target_w: Required output width in pixels.
        target_h: Required output height in pixels.

    Returns:
        Filter string that scales to cover and crops excess pixels without padding.
    """
    return (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={target_w}:{target_h}:(iw-ow)/2:(ih-oh)/2,setsar=1"
    )


def _probe_frame_rate(path: Path) -> str:
    """Return the average frame rate of the primary video stream for ffmpeg.

    Args:
        path: Path to the input video.

    Returns:
        Frame-rate fraction accepted by ffmpeg, such as ``"30/1"``.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    frame_rate = result.stdout.strip()
    if result.returncode != 0 or frame_rate in {"", "0/0"}:
        return "30"
    return frame_rate


def _build_realesrgan_cmd(input_dir: Path, output_dir: Path) -> list[str]:
    """Build the Real-ESRGAN command for animation frames.

    Args:
        input_dir: Directory containing extracted PNG frames.
        output_dir: Destination directory for enhanced PNG frames.

    Returns:
        Command that runs the AnimeVideo-v3 model on the first GPU.
    """
    return [
        str(REAL_ESRGAN_EXECUTABLE),
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-n", "realesr-animevideov3",
        "-s", AI_UPSCALE_SCALE,
        "-f", "png",
        "-g", "0",
    ]


def _enhance_video_with_realesrgan(
    video_path: Path,
    out_path: Path,
    encoder: str,
    duration: float,
) -> Path:
    """Use Real-ESRGAN AnimeVideo-v3 to create a detailed 1080p video source.

    Encodes with closed-GOP, no B-frames, and standard pixel format so that the
    output video can be looped instantly via stream copy without re-encoding.

    Args:
        video_path: Low-resolution source video.
        out_path: Destination for the enhanced 1080p video.
        encoder: Video encoder used when rebuilding enhanced frames.
        duration: Source duration in seconds.

    Returns:
        Enhanced video path, or the original source when AI enhancement is unavailable.
    """
    if not AI_UPSCALE_ENABLED:
        print("      AI upscaler disabled (ACE_STEP_AI_UPSCALE=0) -- using Lanczos.")
        return video_path
    if not REAL_ESRGAN_EXECUTABLE.is_file():
        print("      Real-ESRGAN is unavailable -- using Lanczos fallback.")
        return video_path

    width, height = _probe_resolution(video_path)
    is_portrait = height > width
    target_w, target_h = (1080, 1920) if is_portrait else (1920, 1080)
    if width >= target_w and height >= target_h:
        print(f"      Source video is already {width}x{height} (>= 1080p) -- skipping AI enhancement.")
        return video_path

    frames_dir = out_path.parent / "realesrgan_frames"
    enhanced_frames_dir = out_path.parent / "realesrgan_enhanced_frames"
    frames_dir.mkdir(exist_ok=True)
    enhanced_frames_dir.mkdir(exist_ok=True)

    print("      AI enhancing frames with Real-ESRGAN AnimeVideo-v3 (RTX 4060)...")
    extract_cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-map", "0:v:0",
        "-vsync", "0",
        str(frames_dir / "frame%08d.png"),
    ]
    _run_ffmpeg_with_progress(extract_cmd, duration, "      Extracting video frames")

    result = subprocess.run(
        _build_realesrgan_cmd(frames_dir, enhanced_frames_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Real-ESRGAN enhancement failed:\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )

    width, height = _probe_resolution(video_path)
    is_portrait = height > width
    target_w, target_h = (1080, 1920) if is_portrait else (1920, 1080)
    frame_rate = _probe_frame_rate(video_path)
    rebuild_cmd = [
        "ffmpeg",
        "-y",
        "-framerate", frame_rate,
        "-i", str(enhanced_frames_dir / "frame%08d.png"),
        "-vf", _build_cover_scale_filter(target_w, target_h),
        "-an",
    ]
    if "nvenc" in encoder:
        rebuild_cmd += [
            "-c:v", encoder,
            "-preset", "p5",
            "-tune", NVENC_TUNE,
            "-cq", "15",
            "-rc", "vbr",
            "-b:v", "0",
            "-gpu", "0",
            "-g", "30",
            "-bf", "0",
            "-pix_fmt", "yuv420p",
        ]
    else:
        rebuild_cmd += [
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "15",
            "-threads", str(CPU_THREADS),
            "-g", "30",
            "-bf", "0",
            "-pix_fmt", "yuv420p",
        ]
    rebuild_cmd.append(str(out_path))
    _run_ffmpeg_with_progress(rebuild_cmd, duration, "      Encoding enhanced 1080p video")
    return out_path


def _upscale_video_to_1080p(
    video_path: Path,
    out_path: Path,
    encoder: str,
    duration: float,
) -> Path:
    """Upscale video to 1080p if its resolution is below 1080p.

    Encodes with closed-GOP, no B-frames, and standard pixel format so that the
    output video can be looped instantly via stream copy without re-encoding.

    Args:
        video_path: Source input video file.
        out_path: Destination path for upscaled video if upscaling is needed.
        encoder: Video encoder to use ('h264_nvenc' or 'libx264').
        duration: Video duration in seconds (for progress bar).

    Returns:
        Path to the 1080p video (either *video_path* if already >= 1080p, or *out_path*).
    """
    width, height = _probe_resolution(video_path)
    is_portrait = height > width
    target_w, target_h = (1080, 1920) if is_portrait else (1920, 1080)

    if width >= target_w and height >= target_h:
        print(f"      Source video is already {width}x{height} (>= 1080p) -- skipping upscale.")
        return video_path

    print(f"      Upscaling video: {width}x{height} -> {target_w}x{target_h} (1080p)...")

    vf = _build_cover_scale_filter(target_w, target_h)

    is_nvenc = "nvenc" in encoder
    cmd = [
        "ffmpeg",
        "-y",
        "-threads", str(CPU_THREADS),
        "-i", str(video_path),
        "-vf", vf,
    ]

    if is_nvenc:
        cmd += [
            "-c:v", encoder,
            "-preset", NVENC_PRESET,
            "-tune", NVENC_TUNE,
            "-cq", "19",
            "-rc", "vbr",
            "-b:v", "0",
            "-gpu", "0",
            "-g", "30",
            "-bf", "0",
            "-pix_fmt", "yuv420p",
        ]
    else:
        cmd += [
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-threads", str(CPU_THREADS),
            "-g", "30",
            "-bf", "0",
            "-pix_fmt", "yuv420p",
        ]

    cmd += [
        "-an",
        str(out_path),
    ]

    if duration > 0:
        _run_ffmpeg_with_progress(
            cmd,
            total_seconds=duration,
            desc=f"      Upscaling ({encoder})",
        )
    else:
        _run_ffmpeg(cmd, "Video upscale")

    new_w, new_h = _probe_resolution(out_path)
    print(f"      Upscale complete: {new_w}x{new_h}")
    return out_path


def _find_input_video(input_dir: Path) -> Path:
    """Return the first MP4 found in *input_dir* (alphabetical order).

    Args:
        input_dir: Directory to search.

    Returns:
        Path to the first MP4.

    Raises:
        FileNotFoundError: If no MP4 files exist.
    """
    mp4_files = sorted(input_dir.glob("*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No MP4 files found in: {input_dir}")
    return mp4_files[0]


def _remux_to_ts(video_mp4: Path, out_ts: Path) -> Path:
    """Remux MP4 to MPEG-TS container with Annex-B bitstream without re-encoding.

    MPEG-TS has no index/moov atoms, allowing ffmpeg's ``-stream_loop -1``
    to loop infinitely in stream-copy mode without hanging on virtual index allocation.

    Args:
        video_mp4: Input MP4 video.
        out_ts: Output TS file destination.

    Returns:
        Path to the output TS file.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_mp4),
        "-c:v", "copy",
        "-bsf:v", "h264_mp4toannexb",
        "-an",
        str(out_ts),
    ]
    _run_ffmpeg(cmd, "Remux to TS")
    return out_ts


def _build_video_cmd(
    video_ts: Path,
    audio: Path,
    out_path: Path,
    total_seconds: float,
) -> list[str]:
    """Build the ffmpeg command that loops the TS video and merges audio via stream copy.

    Uses native ``-stream_loop -1`` on MPEG-TS and ``-c:v copy -c:a copy`` so the pre-upscaled 1080p video
    is muxed directly with pre-concatenated AAC audio at disk speed without re-encoding frames or demuxer hangs.

    Args:
        video_ts: MPEG-TS video clip to loop.
        audio: Concatenated AAC audio file.
        out_path: Final output MP4 path.
        total_seconds: Target duration (audio length).

    Returns:
        List of command-line arguments for subprocess.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-threads", str(CPU_THREADS),
        "-stream_loop", "-1",
        "-i", str(video_ts),
        "-i", str(audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "copy",
        "-t", str(total_seconds),
        "-shortest",
        "-movflags", "+faststart",
        str(out_path),
    ]
    return cmd


def _parse_time_str(time_str: str) -> float | None:
    """Parse HH:MM:SS.micro into seconds.

    Args:
        time_str: Timestamp string from ffmpeg (e.g. '01:23:45.67').

    Returns:
        Seconds as a float or None if parsing fails.
    """
    parts = time_str.split(":")
    if len(parts) == 3:
        try:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        except ValueError:
            return None
    return None


def _progress_target(processed_seconds: float, total_seconds: float) -> float:
    """Return a display value that reserves the final percent for ffmpeg cleanup.

    Args:
        processed_seconds: Media duration reported by ffmpeg.
        total_seconds: Expected total media duration.

    Returns:
        Progress-bar value capped below the total until ffmpeg exits successfully.
    """
    return min(max(0.0, processed_seconds), total_seconds * 0.99)


def _run_ffmpeg_with_progress(cmd: list[str], total_seconds: float, desc: str) -> None:
    """Run an ffmpeg command while streaming progress to a tqdm progress bar.

    Uses a bounded ring buffer for stderr to prevent unbounded memory growth on long runs.

    Args:
        cmd: Full ffmpeg command list.
        total_seconds: Target duration in seconds for progress calculation.
        desc: Progress bar description / label.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero return code.
    """
    full_cmd = [cmd[0], "-progress", "pipe:1", "-nostats"] + cmd[1:]
    proc = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )
    stderr_lines: deque[str] = deque(maxlen=100)

    def _drain_stderr() -> None:
        """Consume ffmpeg diagnostics with a bounded ring-buffer to prevent memory leaks."""
        if proc.stderr:
            for line in proc.stderr:
                stderr_lines.append(line)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    progress_total = round(total_seconds, 1)
    last_reported_time = 0.0
    with tqdm(
        total=progress_total,
        desc=desc,
        unit="s",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    ) as pbar:
        speed_str = ""
        fps_str = ""
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if not line and proc.poll() is not None:
                break
            line = line.strip()
            if not line or "=" not in line:
                continue

            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()

            cur_time: float | None = None
            if key == "out_time_us":
                try:
                    cur_time = int(val) / 1_000_000.0
                except ValueError:
                    pass
            elif key == "out_time_ms":
                try:
                    cur_time = int(val) / 1_000.0
                except ValueError:
                    pass
            elif key == "out_time":
                cur_time = _parse_time_str(val)
            elif key == "speed":
                speed_str = val
            elif key == "fps":
                try:
                    if float(val) > 0:
                        fps_str = f"{val}fps"
                except ValueError:
                    pass

            if cur_time is not None:
                last_reported_time = max(last_reported_time, cur_time)
                target = _progress_target(cur_time, progress_total)
                if target > pbar.n:
                    pbar.update(target - pbar.n)

            if key == "progress":
                postfix = []
                if speed_str:
                    postfix.append(f"speed={speed_str}")
                if fps_str:
                    postfix.append(fps_str)
                if last_reported_time >= total_seconds:
                    postfix.append("finalizing")
                if postfix:
                    pbar.set_postfix_str(", ".join(postfix))

        proc.wait()
        stderr_thread.join()
        if proc.returncode == 0 and pbar.n < progress_total:
            pbar.update(progress_total - pbar.n)

    stderr_output = "".join(stderr_lines)
    if proc.returncode != 0:
        raise RuntimeError(f"{desc.strip()} failed (exit code {proc.returncode}):\n{stderr_output.strip()}")


def _run_ffmpeg(cmd: list[str], step_name: str) -> None:
    """Run an ffmpeg command, raising RuntimeError on failure.

    Args:
        cmd: Full ffmpeg command list.
        step_name: Human-readable step name for error messages.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero return code.
    """
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{step_name} failed:\n{proc.stderr.decode()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate parallel probe, watermark clean, upscale, audio concat, GPU loop+merge."""
    start_total_time = time.perf_counter()
    print("=" * 60)
    print("  ACE-Step - Draft Audio -> Video Builder  [GPU-optimised]")
    print("=" * 60)

    _require_ffmpeg()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Detect GPU encoder
    driver = _nvidia_driver_version()
    gpu_encoder = _detect_gpu_encoder()
    if gpu_encoder:
        print(f"  GPU encoder : {gpu_encoder} (RTX 4060 NVENC) -- driver {'.'.join(map(str, driver))}")
    else:
        if driver and driver < _NVENC_MIN_DRIVER:
            print(
                f"  GPU encoder : NVENC SKIPPED (driver {'.'.join(map(str, driver))}, "
                f"need >= {'.'.join(map(str, _NVENC_MIN_DRIVER))})"
            )
            print("  >> Update driver: https://www.nvidia.com/Download/index.aspx")
        else:
            print("  GPU encoder : not detected -- using libx264")
        print("  CPU encoder : libx264 (16 threads, i9-14HX)")

    # 1. Collect MP3s & probe files
    print(f"\n[1/6] Scanning for MP3 files in:\n      {GRADIO_OUTPUTS_DIR}")
    mp3_files = _collect_mp3s(GRADIO_OUTPUTS_DIR)

    # Probe input video and all MP3s in PARALLEL
    input_video = _find_input_video(INPUT_DIR)
    all_probe_targets = mp3_files + [input_video]
    print(f"      Probing {len(all_probe_targets)} files in parallel"
          f" ({min(len(all_probe_targets), CPU_THREADS)} workers)...")
    t0 = time.perf_counter()
    durations = _probe_durations_parallel(all_probe_targets)
    probe_secs = time.perf_counter() - t0

    for i, f in enumerate(mp3_files, 1):
        d = durations[f]
        print(f"      {i:>2}. {f.parent.name}/{f.name}  [{_format_seconds(d)}]")
    print(f"      (probed in {probe_secs:.2f}s)")

    video_duration = durations[input_video]
    total_mp3_duration = sum(durations[f] for f in mp3_files)
    loops_needed = max(0, int(total_mp3_duration / video_duration))
    plays = loops_needed + 1

    encoder = gpu_encoder or "libx264"
    timestamp = int(time.time())
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        meta_cleaned_video = tmp_path / f"meta_clean_{input_video.name}"
        deep_cleaned_video = tmp_path / f"deep_clean_{input_video.name}"
        upscaled_video = tmp_path / f"upscaled_1080p_{input_video.name}"
        enhanced_video = tmp_path / f"enhanced_1080p_{input_video.name}"

        # 2. Strip container metadata (C2PA / XMP / EXIF / atoms)
        print("\n[2/6] Stripping container metadata (C2PA / XMP / provenance atoms)...")
        t_meta = time.perf_counter()
        clean_stage_video = strip_container_metadata(input_video, meta_cleaned_video)
        print(f"      Metadata stripped in {time.perf_counter() - t_meta:.2f}s")

        # 3. Deep AI watermark mitigation
        mode = WATERMARK_CLEAN_MODE.lower()
        print(f"\n[3/6] Mitigating AI watermarks (mode: {mode.upper()})...")
        if mode != "off":
            # Auto-fallback: If Real-ESRGAN will re-render all frames, PARANOID diffusion is redundant
            w, h = _probe_resolution(input_video)
            target_w, target_h = (1080, 1920) if h > w else (1920, 1080)
            esrgan_will_run = (
                AI_UPSCALE_ENABLED
                and REAL_ESRGAN_EXECUTABLE.is_file()
                and (w < target_w or h < target_h)
            )
            if mode == "paranoid" and esrgan_will_run:
                print("      Real-ESRGAN is enabled and will reconstruct all pixels from scratch.")
                print("      Auto-switching PARANOID -> FAST mode to avoid redundant diffusion pass.")
                mode = "fast"

            t_clean = time.perf_counter()
            clean_stage_video = deep_clean_frames(
                video_path=clean_stage_video,
                out_path=deep_cleaned_video,
                encoder=encoder,
                duration=video_duration,
                mode=mode,  # type: ignore[arg-type]
            )
            print(f"      Watermark cleaning finished in {time.perf_counter() - t_clean:.1f}s")
        else:
            print("      Frame-level watermark cleaning disabled (ACE_STEP_CLEAN_MODE=off).")

        # 4. Enhance / upscale input video to 1080p if needed
        print("\n[4/6] Enhancing input video to 1080p (if needed)...")
        t_up = time.perf_counter()
        ai_enhanced_video = _enhance_video_with_realesrgan(
            video_path=clean_stage_video,
            out_path=enhanced_video,
            encoder=encoder,
            duration=video_duration,
        )
        ready_video = _upscale_video_to_1080p(
            video_path=ai_enhanced_video,
            out_path=upscaled_video,
            encoder=encoder,
            duration=video_duration,
        )
        if ready_video != clean_stage_video:
            print(f"      Video preparation finished in {time.perf_counter() - t_up:.1f}s")

        # 5. Concatenate audio
        print(f"\n[5/6] Concatenating {len(mp3_files)} audio file(s) ({_format_seconds(total_mp3_duration)} total)...")
        t1 = time.perf_counter()
        total_duration, concat_audio_path, audio_codec_desc = _concat_audio(
            mp3_files, tmp_path, expected_duration=total_mp3_duration
        )
        print(f"      Done in {time.perf_counter() - t1:.1f}s")

        # 6. Loop video + merge audio in ONE fast stream-copy pass
        final_output = OUTPUT_DIR / f"{timestamp}_final.mp4"
        print(f"\n[6/6] Loop video ({_format_seconds(video_duration)} x~{plays}) + merge audio (stream copy - zero re-encode)")
        print(f"      Source video : {ready_video.name}")
        print("      Video stream : copy (lossless, instant via MPEG-TS)")
        print(f"      Audio stream : copy ({audio_codec_desc})")
        print(f"      Output       : {final_output.name}")

        t2 = time.perf_counter()
        loop_ts = tmp_path / "loop_source.ts"
        _remux_to_ts(ready_video, loop_ts)
        cmd = _build_video_cmd(
            video_ts=loop_ts,
            audio=concat_audio_path,
            out_path=final_output,
            total_seconds=total_duration,
        )

        _run_ffmpeg_with_progress(
            cmd,
            total_seconds=total_duration,
            desc="      Video stream copy & mux",
        )

        encode_secs = time.perf_counter() - t2
        print(f"      Muxed in {encode_secs:.1f}s")

    size_mb = final_output.stat().st_size / 1_048_576
    total_elapsed = time.perf_counter() - start_total_time
    print(
        f"\n[OK] Done in {_format_seconds(total_elapsed)} ({total_elapsed:.1f}s)!  "
        f"({size_mb:.1f} MB)\n    {final_output}"
    )
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
