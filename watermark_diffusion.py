# -*- coding: utf-8 -*-
"""watermark_diffusion.py - SDXL-Turbo + ControlNet Canny diffusion pass (PARANOID mode).

Disrupts invisible pixel watermarks by passing each frame through a single
SDXL-Turbo img2img step at low denoise strength (default 0.10).  At this
strength the UNet re-samples latents at watermarked frequency bands without
visually altering content, overwriting the generator's biased pixel distribution.

VRAM budget (fp16, RTX 4060 8 GB)
----------------------------------
SDXL-Turbo UNet              ~3.8 GB
ControlNet Canny SDXL-1.0    ~1.4 GB
VAE encoder + decoder        ~0.9 GB  (tiled; prevents 1080p OOM)
Frame tensor 1080p fp16      ~0.4 GB
CUDA runtime overhead        ~0.3 GB
Total peak                   ~6.8 GB  (1.2 GB headroom on RTX 4060)

Model downloads (lazy, first PARANOID run only)
-----------------------------------------------
stabilityai/sdxl-turbo               ~6.9 GB
diffusers/controlnet-canny-sdxl-1.0  ~2.5 GB
Cached under ~/.cache/huggingface/hub/

Performance (RTX 4060)
----------------------
~2.8–3.5 s/frame at 1920×1080 with denoise=0.10, 2 inference steps.
10-second source loop (300 frames @30 fps) ≈ 18 minutes.

Auto-downgrade
--------------
deep_clean_frames() in watermark_clean.py catches RuntimeError from this
module and returns the original video path safely when CUDA is unavailable.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from math import isfinite
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Configuration (overridable via environment variables)
# ---------------------------------------------------------------------------
_DENOISE_STRENGTH: float = float(os.environ.get("ACE_STEP_DENOISE", "0.10"))
_DIFFUSION_STEPS: int = 2          # SDXL-Turbo: 1–4 steps recommended
_GUIDANCE_SCALE: float = 0.0       # Turbo uses classifier-free guidance = 0
_CONTROLNET_SCALE: float = 0.55    # ControlNet conditioning strength

_SDXL_TURBO_REPO: str = "stabilityai/sdxl-turbo"
_CONTROLNET_REPO: str = "diffusers/controlnet-canny-sdxl-1.0"
_CANNY_LOW: int = 100
_CANNY_HIGH: int = 200
_FRAME_DIGITS: int = 8
_CPU_THREADS: int = 16


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _effective_denoise_strength(strength: float, num_inference_steps: int) -> float:
    """Return a valid img2img strength that schedules at least one denoising step.

    Diffusers truncates ``num_inference_steps * strength`` to an integer.  A
    strength below ``1 / num_inference_steps`` would otherwise select zero
    timesteps and fail before a frame can be rendered.

    Args:
        strength: Requested img2img denoise strength, from zero to one.
        num_inference_steps: Number of scheduler inference steps.

    Returns:
        The requested strength, raised only when necessary to schedule one step.

    Raises:
        ValueError: If the strength or number of inference steps is invalid.
    """
    if not isfinite(strength) or not 0.0 < strength <= 1.0:
        raise ValueError("ACE_STEP_DENOISE must be greater than 0 and at most 1.")
    if num_inference_steps < 1:
        raise ValueError("Diffusion inference steps must be at least 1.")
    return max(strength, 1.0 / num_inference_steps)


def _load_pipeline():
    """Load SDXL-Turbo + ControlNet Canny with full 8 GB VRAM optimisations.

    Downloads model weights lazily on the first PARANOID-mode run; subsequent
    calls are served from the HuggingFace Hub cache (~/.cache/huggingface/hub/).
    Progress is logged clearly so long downloads are not mistaken for hangs.

    Returns:
        Configured StableDiffusionXLControlNetImg2ImgPipeline in fp16.

    Raises:
        RuntimeError: If CUDA is not available on this system.
    """
    import torch
    from diffusers import (
        ControlNetModel,
        StableDiffusionXLControlNetImg2ImgPipeline,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PARANOID mode requires a CUDA GPU.  "
            "Set ACE_STEP_CLEAN_MODE=fast to use the CPU FFT path instead."
        )

    logger.info(
        "Loading ControlNet Canny SDXL (fp16) — first run downloads ~9.4 GB "
        "to ~/.cache/huggingface/hub/  …"
    )
    controlnet = ControlNetModel.from_pretrained(
        _CONTROLNET_REPO,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        _SDXL_TURBO_REPO,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    # VRAM guards — order matters: offload first, then tile VAE, then slice attention
    pipe.enable_model_cpu_offload()   # hot-swap idle components to system RAM
    pipe.enable_vae_tiling()           # tile VAE decode → no 1080p OOM
    pipe.enable_attention_slicing()    # slice UNet attention → −200 MB peak VRAM
    logger.info("Pipeline ready (fp16 | cpu_offload | vae_tiling | attn_slicing).")
    return pipe


def _run_diffusion_frame(
    pipe,
    frame_rgb: np.ndarray,
    canny_rgb: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Run a single SDXL-Turbo img2img pass on one RGB frame.

    The empty prompt combined with low denoise strength means the model
    re-samples latent noise at watermarked frequency positions while the
    ControlNet Canny map preserves global structure.

    Args:
        pipe: Loaded StableDiffusionXLControlNetImg2ImgPipeline.
        frame_rgb: uint8 (H, W, 3) RGB frame.
        canny_rgb: uint8 (H, W, 3) Canny edge map (same spatial dimensions).
        strength: Valid denoise strength that schedules at least one timestep.

    Returns:
        Cleaned uint8 (H, W, 3) RGB frame.
    """
    import torch
    from PIL import Image

    with torch.inference_mode():
        result = pipe(
            prompt="",
            image=Image.fromarray(frame_rgb),
            control_image=Image.fromarray(canny_rgb),
            strength=strength,
            num_inference_steps=_DIFFUSION_STEPS,
            guidance_scale=_GUIDANCE_SCALE,
            controlnet_conditioning_scale=_CONTROLNET_SCALE,
            output_type="np",
        ).images[0]

    torch.cuda.empty_cache()   # reclaim VRAM budget for the next frame
    return (result * 255).astype(np.uint8)


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
    r = subprocess.run(cmd, capture_output=True, text=True)
    fr = r.stdout.strip()
    return fr if fr and fr != "0/0" else "30"


def _rebuild_video(
    frames_dir: Path,
    out_path: Path,
    frame_rate: str,
    encoder: str,
) -> None:
    """Re-encode cleaned PNG frames into a loop-compatible MP4.

    Uses closed-GOP (``-g 30``) and no B-frames (``-bf 0``) so the output
    is immediately stream-copyable in the final loop+merge step.

    Args:
        frames_dir: Directory of sequential PNGs.
        out_path: Destination MP4 path.
        frame_rate: ffmpeg-compatible rate string.
        encoder: 'h264_nvenc' or 'libx264'.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero return code.
    """
    is_nvenc = "nvenc" in encoder
    cmd = [
        "ffmpeg", "-y",
        "-framerate", frame_rate,
        "-i", f"{frames_dir}/frame%0{_FRAME_DIGITS}d.png",
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


def diffusion_clean(
    video_path: Path,
    out_path: Path,
    encoder: str,
    duration: float,
) -> Path:
    """Run the full diffusion re-render watermark disruption pipeline.

    Extracts lossless PNG frames, processes each through a single SDXL-Turbo
    img2img step with ControlNet Canny conditioning, then rebuilds the video
    with NVENC or libx264 at near-lossless quality (CRF/CQ 16).

    Args:
        video_path: Input video (container metadata already stripped).
        out_path: Cleaned video destination.
        encoder: 'h264_nvenc' or 'libx264'.
        duration: Source duration in seconds (for log output).

    Returns:
        out_path after successful rebuild.

    Raises:
        RuntimeError: If CUDA is unavailable or any pipeline step fails.
    """
    strength = _effective_denoise_strength(_DENOISE_STRENGTH, _DIFFUSION_STEPS)
    if strength != _DENOISE_STRENGTH:
        logger.warning(
            "Raising denoise strength from {} to {} so {} diffusion step(s) render a frame.",
            _DENOISE_STRENGTH,
            strength,
            _DIFFUSION_STEPS,
        )
    pipe = _load_pipeline()

    try:
        with tempfile.TemporaryDirectory(prefix="diff_clean_") as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            clean_dir = tmp_path / "clean"
            raw_dir.mkdir()
            clean_dir.mkdir()

            extract_cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-map", "0:v:0", "-vsync", "0",
                f"{raw_dir}/frame%0{_FRAME_DIGITS}d.png",
            ]
            r = subprocess.run(extract_cmd, capture_output=True)
            if r.returncode != 0:
                raise RuntimeError(
                    f"Frame extraction failed: {r.stderr.decode(errors='replace').strip()}"
                )

            frame_paths = sorted(raw_dir.glob("*.png"))
            if not frame_paths:
                raise RuntimeError("No frames extracted from video.")

            logger.info(
                "Diffusion re-render: {} frames, denoise={}, steps={}, {:.0f}s source",
                len(frame_paths), _DENOISE_STRENGTH, _DIFFUSION_STEPS, duration,
            )

            for i, fp in enumerate(frame_paths):
                bgr = cv2.imread(str(fp))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, threshold1=_CANNY_LOW, threshold2=_CANNY_HIGH)
                canny_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

                cleaned_rgb = _run_diffusion_frame(pipe, rgb, canny_rgb, strength)
                cleaned_bgr = cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(clean_dir / fp.name), cleaned_bgr,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1],
                )
                if (i + 1) % 10 == 0:
                    logger.info("  Rendered {}/{} frames", i + 1, len(frame_paths))

            frame_rate = _probe_frame_rate(video_path)
            _rebuild_video(clean_dir, out_path, frame_rate, encoder)

    finally:
        # Always release GPU memory even if the pipeline errors mid-run.
        import torch
        del pipe
        torch.cuda.empty_cache()

    logger.info("Diffusion clean complete → {}", out_path.name)
    return out_path
