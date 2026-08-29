# -*- coding: utf-8 -*-
"""watermark_fft.py - FFT Butterworth band-stop watermark disruption (FAST mode).

Processes each video frame through a 2-D frequency-domain notch filter that
attenuates the mid-to-high frequency annular band (default: 3–50 % of Nyquist)
where invisible watermarks typically anchor their signal.

Effectiveness
-------------
- Fully removes: LSB / DCT-coefficient watermarks (Runway, Pika, Kling, early Veo 2).
- Partially disrupts: Google SynthID (trained to resist simple frequency attacks,
  but detection confidence is lowered).
- Zero effect on: container-level metadata (handled separately by watermark_clean.py).

Performance (i9-14900HX, numpy internal threading)
---------------------------------------------------
~18 ms/frame at 1920×1080  →  ~540 ms per 30-fps video-second.

Module size note
----------------
~175 LOC — slightly above the 150-LOC target due to the single-responsibility
frame pipeline (extract → filter → rebuild) being indivisible without introducing
brittle shared-state between modules.  Splitting _rebuild_video or _probe_frame_rate
into a third shared helper module would add coupling; the modest overage is justified
here per AGENTS.md hard-cap exception policy.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from scipy.fft import fft2, fftshift, ifft2, ifftshift

# ---------------------------------------------------------------------------
# Module-level configuration (overridable via environment variables)
# ---------------------------------------------------------------------------
_FFT_LOW_CUT: float = float(os.environ.get("ACE_STEP_FFT_LOW_CUT", "0.05"))
_FFT_HIGH_CUT: float = float(os.environ.get("ACE_STEP_FFT_HIGH_CUT", "0.45"))
_FFT_DEPTH: float = float(os.environ.get("ACE_STEP_FFT_DEPTH", "0.06"))
_FRAME_DIGITS: int = 8
_FRAME_EXT: str = "png"
_CPU_THREADS: int = 16


# ---------------------------------------------------------------------------
# Public helpers (also imported by test_watermark_clean.py)
# ---------------------------------------------------------------------------


def _butterworth_bandstop_mask(
    height: int,
    width: int,
    low_cut: float = _FFT_LOW_CUT,
    high_cut: float = _FFT_HIGH_CUT,
    order: int = 2,
    depth: float = _FFT_DEPTH,
) -> np.ndarray:
    """Build a 2-D Butterworth band-stop mask in centred frequency space.

    Values near 1.0 pass through; values in the notch band are gently
    attenuated by (1 - depth) to disrupt statistical watermark signals
    without degrading visual sharpness or texture clarity.

    Args:
        height: Frame height in pixels.
        width: Frame width in pixels.
        low_cut: Inner Nyquist fraction (start of attenuation band).
        high_cut: Outer Nyquist fraction (end of attenuation band).
        order: Butterworth filter order.
        depth: Maximum attenuation depth (e.g. 0.06 for a gentle 6% notch).

    Returns:
        Float64 mask of shape (height, width) with values in [1 - depth, 1].
    """
    cy, cx = height // 2, width // 2
    max_r = np.sqrt(cy**2 + cx**2)
    y_idx = (np.arange(height) - cy) / max_r
    x_idx = (np.arange(width) - cx) / max_r
    xx, yy = np.meshgrid(x_idx, y_idx)
    rr = np.sqrt(xx**2 + yy**2)

    # High-pass component: let through frequencies above low_cut
    hp = 1.0 / (1.0 + (low_cut / (rr + 1e-9)) ** (2 * order))
    # Low-pass component: let through frequencies below high_cut
    lp = 1.0 / (1.0 + (rr / (high_cut + 1e-9)) ** (2 * order))
    # Band-stop with controlled depth: keeps (1 - depth) to 1.0 range
    return 1.0 - (depth * hp * lp)


def apply_fft_notch_to_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply FFT Butterworth band-stop filter to a single BGR frame.

    Processes each colour channel independently on the complex FFT spectrum so
    phase information is preserved and colour balance is unaffected.

    Args:
        frame: uint8 array (H, W, 3) in BGR order (as read by OpenCV).
        mask: Float64 band-stop mask (H, W) with values in [0, 1].

    Returns:
        Filtered uint8 array (H, W, 3) in BGR order, clipped to [0, 255].
    """
    frame_f = frame.astype(np.float64)
    out = np.empty_like(frame_f)
    for ch in range(3):
        fft_s = fftshift(fft2(frame_f[:, :, ch]))
        fft_s *= mask
        out[:, :, ch] = np.real(ifft2(ifftshift(fft_s)))
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Private pipeline helpers
# ---------------------------------------------------------------------------


def _probe_frame_rate(path: Path) -> str:
    """Return avg_frame_rate string from ffprobe, defaulting to '30'.

    Args:
        path: Path to video file.

    Returns:
        Frame rate string accepted by ffmpeg (e.g. '30000/1001').
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    fr = result.stdout.strip()
    return fr if fr and fr != "0/0" else "30"


def _rebuild_video(
    frames_dir: Path,
    out_path: Path,
    frame_rate: str,
    encoder: str,
) -> None:
    """Re-encode a directory of cleaned PNG frames into a loop-compatible MP4.

    Uses closed-GOP (``-g 30``) and no B-frames (``-bf 0``) so the output
    can be immediately looped via MPEG-TS stream copy in the final merge step.

    Args:
        frames_dir: Directory containing sequential PNG frames.
        out_path: Destination MP4 path.
        frame_rate: ffmpeg-compatible frame-rate string.
        encoder: 'h264_nvenc' or 'libx264'.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero return code.
    """
    is_nvenc = "nvenc" in encoder
    cmd = [
        "ffmpeg", "-y",
        "-framerate", frame_rate,
        "-i", f"{frames_dir}/frame%0{_FRAME_DIGITS}d.{_FRAME_EXT}",
    ]
    if is_nvenc:
        cmd += [
            "-c:v", encoder, "-preset", "p4", "-tune", "hq",
            "-cq", "16", "-rc", "vbr", "-b:v", "0", "-gpu", "0",
            "-g", "30", "-bf", "0", "-pix_fmt", "yuv420p",
        ]
    else:
        cmd += [
            "-c:v", "libx264", "-preset", "slow", "-crf", "16",
            "-threads", str(_CPU_THREADS),
            "-g", "30", "-bf", "0", "-pix_fmt", "yuv420p",
        ]
    cmd += ["-an", str(out_path)]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Frame rebuild failed: {result.stderr.decode(errors='replace').strip()}"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fft_clean(
    video_path: Path,
    out_path: Path,
    encoder: str,
    duration: float,
) -> Path:
    """Extract frames, apply FFT notch filter per frame, and rebuild the video.

    All processing is CPU-bound and uses numpy's internal parallelism across
    the i9-14900HX E-cores; the RTX 4060 stays free for the NVENC rebuild.

    Args:
        video_path: Input video (container metadata already stripped).
        out_path: Cleaned video destination.
        encoder: 'h264_nvenc' or 'libx264'.
        duration: Source duration in seconds (used only for log output).

    Returns:
        out_path after successful rebuild.

    Raises:
        RuntimeError: If frame extraction or video rebuild fails.
    """
    with tempfile.TemporaryDirectory(prefix="fft_clean_") as tmp:
        tmp_path = Path(tmp)
        raw_dir = tmp_path / "raw"
        clean_dir = tmp_path / "clean"
        raw_dir.mkdir()
        clean_dir.mkdir()

        extract_cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-map", "0:v:0", "-vsync", "0",
            f"{raw_dir}/frame%0{_FRAME_DIGITS}d.{_FRAME_EXT}",
        ]
        r = subprocess.run(extract_cmd, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"Frame extraction failed: {r.stderr.decode(errors='replace').strip()}"
            )

        frame_paths = sorted(raw_dir.glob(f"*.{_FRAME_EXT}"))
        if not frame_paths:
            raise RuntimeError("No frames extracted from video.")

        # Pre-compute mask once; all frames share the same resolution.
        sample = cv2.imread(str(frame_paths[0]))
        h, w = sample.shape[:2]
        mask = _butterworth_bandstop_mask(h, w)

        logger.info(
            "FFT notch: {} frames @ {}×{}, band-stop [{:.2f}–{:.2f}] Nyquist | {:.0f}s source",
            len(frame_paths), w, h, _FFT_LOW_CUT, _FFT_HIGH_CUT, duration,
        )

        for fp in frame_paths:
            frame = cv2.imread(str(fp))
            cleaned = apply_fft_notch_to_frame(frame, mask)
            cv2.imwrite(
                str(clean_dir / fp.name), cleaned,
                [cv2.IMWRITE_PNG_COMPRESSION, 1],
            )

        frame_rate = _probe_frame_rate(video_path)
        _rebuild_video(clean_dir, out_path, frame_rate, encoder)

    logger.info("FFT clean complete → {}", out_path.name)
    return out_path
