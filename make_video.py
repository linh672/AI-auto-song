# -*- coding: utf-8 -*-
"""make_video.py - Draft all MP3s from gradio_outputs and build a video.

Workflow
--------
1. Scan ``gradio_outputs/`` recursively for every ``*.mp3`` file.
2. Probe all MP3 durations in parallel (ThreadPoolExecutor).
3. Concatenate them (in folder-name order) into a single draft audio track.
4. Loop-extend the first ``*.mp4`` found in ``input/`` to match that duration.
5. Merge the looped video with the concatenated audio and write the result to
   ``output/<timestamp>_final.mp4``.

Hardware targets
----------------
- CPU : Intel i9-14HX  (24 cores / 32 threads)  <- primary encoder
- GPU : RTX 4060 8 GB  (NVENC requires driver >= 570.0 for ffmpeg API 13.0)
- RAM : 32 GB

Encoder selection (auto-detected at runtime)
--------------------------------------------
1. h264_nvenc  -- RTX 4060 NVENC (requires NVIDIA driver >= 570.0)
2. libx264     -- CPU fallback, preset=ultrafast (fast enough: ~25s for 9 min)

To unlock GPU encoding, update your NVIDIA driver:
  https://www.nvidia.com/Download/index.aspx

Optimisations applied
----------------------
- Parallel ffprobe calls (ThreadPoolExecutor, up to CPU_THREADS workers).
- Loop + merge in a single ffmpeg pass (no intermediate looped video file).
- NVENC pre-check via ``nvidia-smi`` driver version before attempting encode.
- ``-threads`` tuned for i9-14HX P-core count.
- Audio concat decodes MP3s once to PCM -- no intermediate transcode.

Dependencies
------------
Only the standard library and **ffmpeg >= 6.0** (must be on PATH) are required.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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

# AAC audio bitrate for the final output
AUDIO_BITRATE = "192k"

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


def _concat_audio(mp3_files: list[Path], out_path: Path) -> float:
    """Concatenate *mp3_files* into a single WAV file at *out_path*.

    Uses ffmpeg concat demuxer with ``-threads`` set to ``CPU_THREADS`` so the
    MP3 decoder can use multiple threads for the decode pass.

    Args:
        mp3_files: Ordered list of MP3 paths to concatenate.
        out_path: Destination WAV file path.

    Returns:
        Total duration of the concatenated audio in seconds.
    """
    list_file = out_path.with_suffix(".txt")
    with list_file.open("w", encoding="utf-8") as fh:
        for mp3 in mp3_files:
            safe = str(mp3).replace("\\", "/")
            fh.write(f"file '{safe}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-threads", str(CPU_THREADS),
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-ar", "44100",
        "-ac", "2",
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    print(f"  [ffmpeg] Concatenating {len(mp3_files)} MP3(s) -> {out_path.name}")
    proc = subprocess.run(cmd, capture_output=True)
    list_file.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed:\n{proc.stderr.decode()}")

    duration = _probe_duration(out_path)
    mins, secs = divmod(duration, 60)
    print(f"  Total audio duration: {int(mins)}m {secs:.1f}s  ({duration:.2f}s)")
    return duration


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


def _build_video_cmd(
    video: Path,
    audio: Path,
    out_path: Path,
    total_seconds: float,
    video_duration: float,
    encoder: str,
) -> list[str]:
    """Build the ffmpeg command that loops the video and merges the audio in one pass.

    Combines the loop + merge into a single ffmpeg invocation to avoid writing
    an intermediate looped file, saving both disk I/O and time.

    Args:
        video: Source video file.
        audio: Concatenated audio WAV file.
        out_path: Final output MP4 path.
        total_seconds: Target duration (audio length).
        video_duration: Duration of the source video clip.
        encoder: Video encoder to use (e.g. ``'h264_nvenc'`` or ``'libx264'``).

    Returns:
        List of command-line arguments for subprocess.
    """
    loops_needed = max(0, int(total_seconds / video_duration))
    is_nvenc = "nvenc" in encoder

    cmd = [
        "ffmpeg",
        "-y",
        "-threads", str(CPU_THREADS),
        # Loop the video input
        "-stream_loop", str(loops_needed),
        "-i", str(video),
        # Audio input
        "-i", str(audio),
        # Trim to exact audio length
        "-t", str(total_seconds),
    ]

    if is_nvenc:
        # GPU encode: upload decoded frames to NVENC on RTX 4060
        cmd += [
            "-c:v", encoder,
            "-preset", NVENC_PRESET,
            "-tune", NVENC_TUNE,
            "-cq", NVENC_CQ,
            "-rc", "vbr",
            "-b:v", "0",
            # GPU decode hint (speeds up demux on NVENC path)
            "-gpu", "0",
        ]
    else:
        # CPU fallback: libx264 ultrafast
        cmd += [
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-threads", str(CPU_THREADS),
        ]

    cmd += [
        "-c:a", "aac",
        "-b:a", AUDIO_BITRATE,
        "-shortest",
        str(out_path),
    ]
    return cmd


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
    """Orchestrate parallel probe, audio concat, GPU loop+merge into a final video."""
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
            print("  GPU encoder : not detected -- using libx264 ultrafast")
        print("  CPU encoder : libx264 ultrafast (16 threads, i9-14HX)")

    # 1. Collect MP3s
    print(f"\n[1/3] Scanning for MP3 files in:\n      {GRADIO_OUTPUTS_DIR}")
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
        m, s = divmod(d, 60)
        print(f"      {i:>2}. {f.parent.name}/{f.name}  [{int(m)}m{s:.0f}s]")
    print(f"      (probed in {probe_secs:.2f}s)")

    video_duration = durations[input_video]
    total_mp3_duration = sum(durations[f] for f in mp3_files)
    loops_needed = max(0, int(total_mp3_duration / video_duration))
    plays = loops_needed + 1

    # 2. Concatenate audio
    timestamp = int(time.time())
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        concat_wav = tmp_path / "concat_audio.wav"

        print(f"\n[2/3] Concatenating audio...")
        t1 = time.perf_counter()
        total_duration = _concat_audio(mp3_files, concat_wav)
        print(f"      Done in {time.perf_counter() - t1:.1f}s")

        # 3. Loop video + merge audio in ONE ffmpeg pass (no intermediate file)
        final_output = OUTPUT_DIR / f"{timestamp}_final.mp4"
        encoder = gpu_encoder or "libx264"
        print(f"\n[3/3] Loop video ({video_duration:.1f}s x~{plays}) + merge audio -> GPU encode")
        print(f"      Source video : {input_video.name}")
        print(f"      Encoder      : {encoder}")
        print(f"      Output       : {final_output.name}")

        t2 = time.perf_counter()
        cmd = _build_video_cmd(
            video=input_video,
            audio=concat_wav,
            out_path=final_output,
            total_seconds=total_duration,
            video_duration=video_duration,
            encoder=encoder,
        )

        _run_ffmpeg(cmd, f"loop+merge ({encoder})")

        encode_secs = time.perf_counter() - t2
        print(f"      Encoded in {encode_secs:.1f}s")

    size_mb = final_output.stat().st_size / 1_048_576
    print(f"\n[OK] Done!  ({size_mb:.1f} MB)\n    {final_output}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
