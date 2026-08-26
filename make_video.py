# -*- coding: utf-8 -*-
"""make_video.py - Draft all MP3s from gradio_outputs and build a video.

Workflow
--------
1. Scan ``gradio_outputs/`` recursively for every ``*.mp3`` file.
2. Probe all MP3 durations and input video resolution in parallel.
3. Upscale the input video in ``input/`` to 1080p if below 1080p.
4. Concatenate MP3s (in folder-name order) into a single draft audio track.
5. Loop-extend the 1080p video to match that duration.
6. Merge the looped video with the concatenated audio and write the result to
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
- Upscale the short input video once before looping using NVENC.
- Stream-copy looped video (-c:v copy with -stream_loop -1) to mux 4+ hours in seconds without re-encoding.
- NVENC pre-check via ``nvidia-smi`` driver version before attempting upscale encode.
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

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

# AAC audio quality: VBR mode, 0 = best (~256-320 kbps), 2 = good (~160 kbps).
# Use -q:a (VBR) instead of -b:a (CBR) for better quality/size ratio.
AUDIO_QUALITY = "0"

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


def _concat_audio(
    mp3_files: list[Path],
    out_path: Path,
    expected_duration: float | None = None,
) -> float:
    """Concatenate *mp3_files* into a single WAV file at *out_path*.

    Uses ffmpeg concat demuxer with ``-threads`` set to ``CPU_THREADS`` so the
    MP3 decoder can use multiple threads for the decode pass.

    Args:
        mp3_files: Ordered list of MP3 paths to concatenate.
        out_path: Destination WAV file path.
        expected_duration: Optional total expected duration in seconds for progress bar.

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
    try:
        if expected_duration and expected_duration > 0:
            _run_ffmpeg_with_progress(
                cmd,
                total_seconds=expected_duration,
                desc="      Audio concat",
            )
        else:
            _run_ffmpeg(cmd, "Audio concat")
    finally:
        list_file.unlink(missing_ok=True)

    duration = _probe_duration(out_path)
    print(f"      Total audio duration: {_format_seconds(duration)}  ({duration:.2f}s)")
    return duration


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


def _upscale_video_to_1080p(
    video_path: Path,
    out_path: Path,
    encoder: str,
    duration: float,
) -> Path:
    """Upscale video to 1080p if its resolution is below 1080p.

    If the video is already 1080p (or higher), returns *video_path* unchanged.
    Otherwise, scales the video to 1080p (1920x1080 for landscape, 1080x1920 for portrait)
    using high-quality Lanczos resampling and a centered crop to fill the frame, saving to *out_path*.

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
        ]
    else:
        cmd += [
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-threads", str(CPU_THREADS),
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
    """Remux MP4 to MPEG-TS container without re-encoding.

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

    Uses native ``-stream_loop -1`` on MPEG-TS and ``-c:v copy`` so the pre-upscaled 1080p video
    is muxed directly with AAC audio at disk speed without re-encoding frames or demuxer hangs.

    Args:
        video_ts: MPEG-TS video clip to loop.
        audio: Concatenated audio WAV file.
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
        "-c:a", "aac",
        "-q:a", AUDIO_QUALITY,
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
    stderr_parts: list[str] = []

    def _drain_stderr() -> None:
        """Consume ffmpeg diagnostics so its stderr pipe cannot block the process."""
        if proc.stderr:
            stderr_parts.append(proc.stderr.read())

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

    stderr_output = "".join(stderr_parts)
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
    """Orchestrate parallel probe, upscale input video to 1080p, audio concat, GPU loop+merge into a final video."""
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
            print("  GPU encoder : not detected -- using libx264 ultrafast")
        print("  CPU encoder : libx264 ultrafast (16 threads, i9-14HX)")

    # 1. Collect MP3s & probe files
    print(f"\n[1/4] Scanning for MP3 files in:\n      {GRADIO_OUTPUTS_DIR}")
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
        upscaled_video = tmp_path / f"upscaled_1080p_{input_video.name}"
        concat_wav = tmp_path / "concat_audio.wav"

        # 2. Upscale input video to 1080p if needed
        print("\n[2/4] Upscaling input video to 1080p (if needed)...")
        t_up = time.perf_counter()
        ready_video = _upscale_video_to_1080p(
            video_path=input_video,
            out_path=upscaled_video,
            encoder=encoder,
            duration=video_duration,
        )
        if ready_video != input_video:
            print(f"      Upscaling finished in {time.perf_counter() - t_up:.1f}s")

        # 3. Concatenate audio
        print(f"\n[3/4] Concatenating {len(mp3_files)} audio file(s) ({_format_seconds(total_mp3_duration)} total)...")
        t1 = time.perf_counter()
        total_duration = _concat_audio(mp3_files, concat_wav, expected_duration=total_mp3_duration)
        print(f"      Done in {time.perf_counter() - t1:.1f}s")

        # 4. Loop video + merge audio in ONE fast stream-copy pass
        final_output = OUTPUT_DIR / f"{timestamp}_final.mp4"
        print(f"\n[4/4] Loop video ({_format_seconds(video_duration)} x~{plays}) + merge audio (stream copy - zero re-encode)")
        print(f"      Source video : {ready_video.name}")
        print(f"      Video stream : copy (lossless, instant via MPEG-TS)")
        print(f"      Audio codec  : aac (VBR q={AUDIO_QUALITY})")
        print(f"      Output       : {final_output.name}")

        t2 = time.perf_counter()
        # Convert short video to TS container to avoid MP4 moov index pre-allocation hang
        loop_ts = tmp_path / "loop_source.ts"
        _remux_to_ts(ready_video, loop_ts)

        cmd = _build_video_cmd(
            video_ts=loop_ts,
            audio=concat_wav,
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

