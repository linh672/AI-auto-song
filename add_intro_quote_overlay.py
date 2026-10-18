# -*- coding: utf-8 -*-
"""add_intro_quote_overlay.py - Applies typewriter intro text overlay video by video with strict memory management."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from loguru import logger

from intro_quote_renderer import (
    TypewriterConfig,
    build_lines_layout,
    parse_timeline_intro,
    stream_typewriter_rawvideo,
)


def find_target_videos(archive_dir: Path) -> list[tuple[Path, Path, str, str, str, str]]:
    """Scan archive directory for folders containing timeline.txt and matching final videos."""
    targets = []
    if not archive_dir.is_dir():
        return targets
    for batch_dir in sorted(archive_dir.iterdir()):
        timeline_path = batch_dir / "timeline.txt"
        if not batch_dir.is_dir() or not timeline_path.is_file():
            continue
        title, text1, greeting, quote = parse_timeline_intro(
            timeline_path.read_text(encoding="utf-8", errors="replace")
        )
        expected = batch_dir / f"{title}.mp4"
        candidate = expected if expected.is_file() else next(
            (f for f in batch_dir.glob("*.mp4") if f.stat().st_size > 100_000_000 and "chill dog" in f.name.lower()),
            None,
        )
        if candidate and candidate.is_file():
            targets.append((batch_dir, candidate, title, text1, greeting, quote))
    return targets


def probe_video_timescale_and_fps(path: Path) -> tuple[str, str]:
    """Return (timescale, fps) for video stream via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=time_base,r_frame_rate",
        "-of", "json",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    timescale = "90000"
    fps = "24"
    if res.returncode == 0 and res.stdout.strip():
        try:
            data = json.loads(res.stdout)
            st = data.get("streams", [{}])[0]
            tb = st.get("time_base", "1/90000")
            timescale = tb.split("/")[1] if "/" in tb else "90000"
            fps_raw = st.get("r_frame_rate", "24/1")
            fps = str(int(fps_raw.split("/")[0]) // int(fps_raw.split("/")[1])) if "/" in fps_raw else fps_raw
        except Exception:
            pass
    return timescale, fps


def render_intro_segment(
    video_path: Path,
    out_path: Path,
    layout: list,
    config: TypewriterConfig,
    split_seconds: float = 24.0,
    timescale: str = "90000",
    fps: str = "24",
) -> None:
    """Render the initial spliced video segment with typewriter overlay."""
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-t", str(split_seconds),
        "-i", str(video_path),
        "-f", "rawvideo", "-pix_fmt", "rgba",
        "-s", f"{config.width}x{config.height}", "-r", str(fps), "-i", "pipe:0",
        "-filter_complex", f"[0:v][1:v]overlay=0:0:enable='between(t,0,{config.total_duration})'",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-g", "30", "-bf", "0", "-pix_fmt", "yuv420p",
        "-video_track_timescale", str(timescale), "-c:a", "copy",
        str(out_path),
    ]
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr_file)
        if proc.stdin:
            stream_typewriter_rawvideo(layout, config, proc.stdin)
            proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            stderr_file.seek(0)
            raise RuntimeError(f"Intro render failed:\n{stderr_file.read()}")


def splice_video_with_intro(
    video_path: Path,
    out_path: Path,
    layout: list,
    config: TypewriterConfig,
    split_seconds: float = 24.0,
) -> None:
    """Splice intro segment with source video using inpoint to avoid 26GB intermediate copies."""
    timescale, fps = probe_video_timescale_and_fps(video_path)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        seg1 = tmp / "seg1.mp4"
        list_file = tmp / "concat.txt"
        logger.info(f"Rendering intro segment (0s to {split_seconds}s, timescale={timescale}, fps={fps})...")
        render_intro_segment(video_path, seg1, layout, config, split_seconds, timescale=timescale, fps=fps)

        p1 = str(seg1.resolve()).replace("\\", "/")
        p_src = str(video_path.resolve()).replace("\\", "/")
        list_file.write_text(f"file '{p1}'\nfile '{p_src}'\ninpoint {split_seconds:.1f}\n", encoding="utf-8")

        logger.info("Direct lossless concat stream-copy without duplicating source video...")
        cmd_concat = [
            "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy",
            "-video_track_timescale", str(timescale), "-movflags", "+faststart",
            str(out_path),
        ]
        subprocess.run(cmd_concat, check=True)



def process_single_target(
    target: tuple[Path, Path, str, str, str, str],
    config: TypewriterConfig,
    split_time: float,
    in_place: bool,
) -> None:
    """Process a single video completely and release all temporary resources immediately."""
    _, vid_path, title, t1, gr, q = target
    logger.info(f"\nProcessing [{vid_path.name}]")
    layout = build_lines_layout(t1, gr, q, config)
    tmp_out = vid_path.parent / f"{vid_path.stem}.tmp_intro.mp4"
    final_target = vid_path if in_place else vid_path.parent / f"{vid_path.stem}_with_intro.mp4"

    try:
        splice_video_with_intro(vid_path, tmp_out, layout, config, split_time)
        if not tmp_out.is_file() or tmp_out.stat().st_size < 100_000_000:
            raise RuntimeError(f"Output video validation failed for {vid_path.name}")
        if in_place:
            backup = vid_path.parent / f"{vid_path.stem}.bak"
            shutil.move(str(vid_path), str(backup))
            shutil.move(str(tmp_out), str(vid_path))
            backup.unlink(missing_ok=True)
            logger.info(f"Updated in-place: {vid_path.name} ({vid_path.stat().st_size / 1_048_576:.1f} MB)")
        else:
            if final_target.exists():
                final_target.unlink()
            shutil.move(str(tmp_out), str(final_target))
            logger.info(f"Created: {final_target.name} ({final_target.stat().st_size / 1_048_576:.1f} MB)")
    finally:
        if tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
        gc.collect()



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for intro quote overlay CLI."""
    p = argparse.ArgumentParser(description="Add typewriter intro text to chill videos.")
    p.add_argument("--archive-dir", type=Path, default=Path("archive"), help="Archive folder.")
    p.add_argument("--batch", type=str, default=None, help="Process only specific batch folder.")
    p.add_argument("--duration", type=float, default=20.0, help="Intro active seconds (20s).")
    p.add_argument("--anim-duration", type=float, default=2.5, help="Typewriter in/out seconds (2.5s).")
    p.add_argument("--split-time", type=float, default=24.0, help="Keyframe splice time (24s).")
    p.add_argument("--in-place", action="store_true", help="Replace original video file in-place.")
    p.add_argument("--dry-run", action="store_true", help="Show targets without rendering.")
    return p.parse_args()


def main() -> None:
    """CLI entrypoint to discover and apply typewriter intro quote overlays video by video."""
    args = parse_args()
    config = TypewriterConfig(total_duration=args.duration, anim_duration=args.anim_duration)
    targets = find_target_videos(args.archive_dir)
    if args.batch:
        b = args.batch.lower()
        targets = [t for t in targets if b in t[0].name.lower() or b in t[1].name.lower()]
    logger.info(f"Discovered {len(targets)} video(s) to process:")
    for _, vid_path, title, t1, gr, q in targets:
        logger.info(f" - [{vid_path.name}] Text 1: {t1} | Text 2: {gr} | Quote: {q}")
    if not args.dry_run:
        for target in targets:
            process_single_target(target, config, args.split_time, args.in_place)
        logger.info("All videos processed successfully video-by-video!")
if __name__ == "__main__":
    main()
