"""Worker execution logic for queued generation tasks."""

import traceback
from typing import Any
from loguru import logger

from acestep.queue.task_model import GenerationTask


def apply_task_lora(dit_handler: Any, lora_path: str | None, lora_scale: float = 1.0) -> None:
    """Apply or unload LoRA adapter for a specific task.

    Args:
        dit_handler: DiT handler instance.
        lora_path: Path to LoRA checkpoint or weights, or None to use base model.
        lora_scale: Scale factor for LoRA adapter.
    """
    if not dit_handler or not hasattr(dit_handler, "model") or dit_handler.model is None:
        return

    current_loaded = getattr(dit_handler, "lora_loaded", False)
    clean_path = lora_path.strip() if isinstance(lora_path, str) and lora_path.strip() else None

    if clean_path:
        logger.info(f"[TaskQueue] Loading LoRA for task: {clean_path} (scale={lora_scale})")
        msg = dit_handler.load_lora(clean_path)
        logger.info(f"[TaskQueue] LoRA load response: {msg}")
        if hasattr(dit_handler, "set_lora_scale"):
            dit_handler.set_lora_scale(lora_scale)
    elif current_loaded:
        logger.info("[TaskQueue] Task requests base model. Unloading active LoRA.")
        msg = dit_handler.unload_lora()
        logger.info(f"[TaskQueue] LoRA unload response: {msg}")


def execute_task(
    task: GenerationTask,
    dit_handler: Any,
    llm_handler: Any,
) -> None:
    """Execute a single generation task end-to-end.

    Args:
        task: The GenerationTask to execute.
        dit_handler: DiT handler instance.
        llm_handler: LLM handler instance.
    """
    task.status = "running"
    task.progress = 0.05
    task.status_message = "Setting up LoRA..."

    try:
        from acestep.ui.gradio.events.results.generation_progress import generate_with_progress

        apply_task_lora(dit_handler, task.lora_path, task.lora_scale)
        task.status_message = "Generating audio..."
        task.progress = 0.15

        params = task.params.copy()
        
        generator = generate_with_progress(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            captions=params.get("captions", ""),
            lyrics=params.get("lyrics", ""),
            bpm=params.get("bpm"),
            key_scale=params.get("key_scale"),
            time_signature=params.get("time_signature"),
            vocal_language=params.get("vocal_language"),
            inference_steps=params.get("inference_steps", 25),
            guidance_scale=params.get("guidance_scale", 7.0),
            random_seed_checkbox=params.get("random_seed_checkbox", True),
            seed=params.get("seed", ""),
            reference_audio=params.get("reference_audio"),
            audio_duration=params.get("audio_duration", 30),
            batch_size_input=params.get("batch_size_input", 1),
            src_audio=params.get("src_audio"),
            text2music_audio_code_string=params.get("text2music_audio_code_string", ""),
            repainting_start=params.get("repainting_start", 0.0),
            repainting_end=params.get("repainting_end", 0.0),
            instruction_display_gen=params.get("instruction_display_gen", ""),
            audio_cover_strength=params.get("audio_cover_strength", 0.5),
            cover_noise_strength=params.get("cover_noise_strength", 0.0),
            task_type=params.get("task_type", "text2music"),
            no_fsq=params.get("no_fsq", False),
            use_adg=params.get("use_adg", False),
            cfg_interval_start=params.get("cfg_interval_start", 0.0),
            cfg_interval_end=params.get("cfg_interval_end", 1.0),
            shift=params.get("shift", 1.0),
            infer_method=params.get("infer_method", "ode"),
            sampler_mode=params.get("sampler_mode", "euler"),
            velocity_norm_threshold=params.get("velocity_norm_threshold", 0.0),
            velocity_ema_factor=params.get("velocity_ema_factor", 0.0),
            dcw_enabled=params.get("dcw_enabled", True),
            dcw_mode=params.get("dcw_mode", "double"),
            dcw_scaler=params.get("dcw_scaler", 0.05),
            dcw_high_scaler=params.get("dcw_high_scaler", 0.02),
            dcw_wavelet=params.get("dcw_wavelet", "haar"),
            custom_timesteps=params.get("custom_timesteps", ""),
            audio_format=params.get("audio_format", "mp3"),
            mp3_bitrate=params.get("mp3_bitrate", "320k"),
            mp3_sample_rate=params.get("mp3_sample_rate", 48000),
            lm_temperature=params.get("lm_temperature", 0.85),
            think_checkbox=params.get("think_checkbox", False),
            lm_cfg_scale=params.get("lm_cfg_scale", 2.0),
            lm_top_k=params.get("lm_top_k", 0),
            lm_top_p=params.get("lm_top_p", 0.9),
            lm_negative_prompt=params.get("lm_negative_prompt", "NO USER INPUT"),
            use_cot_metas=params.get("use_cot_metas", True),
            use_cot_caption=params.get("use_cot_caption", False),
            use_cot_language=params.get("use_cot_language", True),
            is_format_caption=params.get("is_format_caption", False),
            constrained_decoding_debug=params.get("constrained_decoding_debug", False),
            allow_lm_batch=params.get("allow_lm_batch", True),
            auto_score=params.get("auto_score", False),
            auto_lrc=params.get("auto_lrc", False),
            score_scale=params.get("score_scale", 0.1),
            lm_batch_chunk_size=params.get("lm_batch_chunk_size", 8),
            enable_normalization=params.get("enable_normalization", False),
            normalization_db=params.get("normalization_db", -14.0),
            fade_in_duration=params.get("fade_in_duration", 0.0),
            fade_out_duration=params.get("fade_out_duration", 0.0),
            latent_shift=params.get("latent_shift", 0.0),
            latent_rescale=params.get("latent_rescale", 1.0),
            repaint_mode=params.get("repaint_mode", "balanced"),
            repaint_strength=params.get("repaint_strength", 0.5),
            retake_variance=params.get("retake_variance", 0.0),
            retake_seed=params.get("retake_seed", ""),
        )

        final_result = None
        for update in generator:
            final_result = update

        if final_result is None:
            raise RuntimeError("Generation did not produce any result")

        all_paths = final_result[8] if len(final_result) > 8 else []
        task.output_audio_paths = [
            path
            for path in (all_paths or [])
            if isinstance(path, str) and not path.endswith(".json")
        ]
        task.generation_info = str(final_result[9]) if len(final_result) > 9 else ""
        generation_status = str(final_result[10]) if len(final_result) > 10 else ""
        if not task.output_audio_paths:
            error_detail = generation_status or "Generation completed without producing audio."
            raise RuntimeError(error_detail)

        task.status = "completed"
        task.progress = 1.0
        task.status_message = "Generation complete!"
        logger.info(f"[TaskQueue] Task {task.id} finished successfully with {len(task.output_audio_paths)} audios")

    except Exception as exc:
        task.status = "failed"
        task.progress = 1.0
        task.error_message = str(exc)
        task.status_message = f"Error: {str(exc)}"
        logger.error(f"[TaskQueue] Task {task.id} failed: {exc}\n{traceback.format_exc()}")
