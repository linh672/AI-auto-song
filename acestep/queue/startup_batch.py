"""Startup batch enqueueing helper for ACE-Step pipeline."""

from __future__ import annotations

from typing import Any
from loguru import logger

from acestep.queue.task_queue_manager import get_task_queue_manager
from acestep.ui.gradio.i18n import get_i18n


def build_default_batch_params(
    dit_handler: Any,
    llm_handler: Any,
    batch_size: int | None = None,
    caption: str | None = None,
    lyrics: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build the default parameter dictionary and title for queued tasks.

    Matches the defaults configured in the Gradio generation interface.

    Args:
        dit_handler: Active DiT handler instance.
        llm_handler: Active LLM handler instance.
        batch_size: Batch size override for parallel audio generation.
        caption: Custom caption text or None to use default placeholder.
        lyrics: Custom lyrics text or empty string.

    Returns:
        Tuple of (title, params_dict).
    """
    i18n = get_i18n()
    default_caption = (caption or "").strip()
    if not default_caption:
        default_caption = i18n.t("generation.caption_placeholder")

    default_lyrics = (lyrics or "").strip()
    title = (default_caption or default_lyrics or "Untitled Song").strip().replace("\n", " ")
    if len(title) > 40:
        title = title[:37] + "..."

    is_turbo = True
    if dit_handler is not None and hasattr(dit_handler, "is_turbo_model"):
        try:
            is_turbo = dit_handler.is_turbo_model()
        except Exception:
            is_turbo = True

    llm_ready = False
    if llm_handler is not None and hasattr(llm_handler, "llm_initialized"):
        llm_ready = bool(llm_handler.llm_initialized)

    params: dict[str, Any] = {
        "captions": default_caption,
        "lyrics": default_lyrics,
        "bpm": None,
        "key_scale": "",
        "time_signature": "",
        "vocal_language": "unknown",
        "inference_steps": 8 if is_turbo else 25,
        "guidance_scale": 1.0 if is_turbo else 7.0,
        "random_seed_checkbox": True,
        "seed": "",
        "reference_audio": None,
        "audio_duration": -1,
        "batch_size_input": batch_size if batch_size is not None else 1,
        "src_audio": None,
        "text2music_audio_code_string": "",
        "repainting_start": 0.0,
        "repainting_end": 0.0,
        "instruction_display_gen": "",
        "audio_cover_strength": 1.0,
        "cover_noise_strength": 0.0,
        "task_type": "text2music",
        "no_fsq": False,
        "use_adg": False,
        "cfg_interval_start": 0.0,
        "cfg_interval_end": 1.0,
        "shift": 3.0,
        "infer_method": "ode",
        "sampler_mode": "euler",
        "velocity_norm_threshold": 0.0,
        "velocity_ema_factor": 0.0,
        "dcw_enabled": True,
        "dcw_mode": "double",
        "dcw_scaler": 0.3 if llm_ready else 0.05,
        "dcw_high_scaler": 0.15 if llm_ready else 0.02,
        "dcw_wavelet": "sym4",
        "custom_timesteps": "",
        "audio_format": "mp3",
        "mp3_bitrate": "320k",
        "mp3_sample_rate": 44100,
        "lm_temperature": 0.85,
        "think_checkbox": llm_ready,
        "lm_cfg_scale": 2.0,
        "lm_top_k": 0,
        "lm_top_p": 0.9,
        "lm_negative_prompt": "",
        "use_cot_metas": True,
        "use_cot_caption": True,
        "use_cot_language": True,
        "is_format_caption": False,
        "constrained_decoding_debug": False,
        "allow_lm_batch": True,
        "auto_score": False,
        "auto_lrc": False,
        "score_scale": 0.1,
        "lm_batch_chunk_size": 8,
        "enable_normalization": True,
        "normalization_db": -14.0,
        "fade_in_duration": 0.5,
        "fade_out_duration": 1.5,
        "latent_shift": 0.0,
        "latent_rescale": 1.0,
        "repaint_mode": "narrow",
        "repaint_strength": 0.5,
        "retake_variance": 0.0,
        "retake_seed": "",
    }
    return title, params


def enqueue_startup_batch_tasks(
    count: int,
    dit_handler: Any,
    llm_handler: Any,
    batch_size: int | None = None,
    caption: str | None = None,
    lyrics: str | None = None,
) -> list[Any]:
    """Enqueue batch generation tasks at application startup.

    Args:
        count: Number of tasks to enqueue (>= 1).
        dit_handler: Active DiT handler instance.
        llm_handler: Active LLM handler instance.
        batch_size: Batch size override for parallel audio generation.
        caption: Custom caption text.
        lyrics: Custom lyrics text.

    Returns:
        List of created GenerationTask instances.
    """
    count = max(1, count)
    qm = get_task_queue_manager()
    qm.initialize_handlers(dit_handler, llm_handler)

    title, params = build_default_batch_params(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        batch_size=batch_size,
        caption=caption,
        lyrics=lyrics,
    )

    tasks = qm.add_tasks(
        title=title,
        params=params,
        count=count,
        lora_path=None,
        lora_scale=1.0,
    )
    logger.info(f"[Batch Mode] Enqueued {len(tasks)} generation tasks: '{title}'")
    print(f"\n{'=' * 60}")
    print(f"Batch Mode Active: Enqueued {len(tasks)} task(s) for generation")
    print(f"Prompt: {title}")
    print(f"{'=' * 60}\n")
    return tasks
