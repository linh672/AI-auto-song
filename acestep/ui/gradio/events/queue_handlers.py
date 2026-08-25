"""Event handlers for the generation task queue tab."""

from typing import Any
import gradio as gr
from loguru import logger

from acestep.queue.task_queue_manager import get_task_queue_manager
from acestep.ui.gradio.i18n import t


def add_to_queue_handler(
    captions: str, lyrics: str, bpm: Any, key_scale: str, time_signature: str,
    vocal_language: str, inference_steps: int, guidance_scale: float,
    random_seed_checkbox: bool, seed: str, reference_audio: Any,
    audio_duration: float, batch_size_input: int, src_audio: Any,
    text2music_audio_code_string: str, repainting_start: float, repainting_end: float,
    instruction_display_gen: str, audio_cover_strength: float,
    cover_noise_strength: float, task_type: str, no_fsq: bool, use_adg: bool,
    cfg_interval_start: float, cfg_interval_end: float, shift: float,
    infer_method: str, sampler_mode: str, velocity_norm_threshold: float,
    velocity_ema_factor: float, dcw_enabled: bool, dcw_mode: str, dcw_scaler: float,
    dcw_high_scaler: float, dcw_wavelet: str, custom_timesteps: str,
    audio_format: str, mp3_bitrate: str, mp3_sample_rate: int,
    lm_temperature: float, think_checkbox: bool, lm_cfg_scale: float,
    lm_top_k: int, lm_top_p: float, lm_negative_prompt: str,
    use_cot_metas: bool, use_cot_caption: bool, use_cot_language: bool,
    is_format_caption: bool, constrained_decoding_debug: bool,
    allow_lm_batch: bool, auto_score: bool, auto_lrc: bool,
    score_scale: float, lm_batch_chunk_size: int,
    enable_normalization: bool, normalization_db: float,
    fade_in_duration: float, fade_out_duration: float,
    latent_shift: float, latent_rescale: float,
    repaint_mode: str, repaint_strength: float,
    retake_variance: float, retake_seed: str,
    lora_path: str, use_lora: bool, lora_scale: float,
) -> tuple[Any, ...]:
    """Capture current UI parameters and enqueue a new generation task."""
    title = (captions or lyrics or "Untitled Song").strip().replace("\n", " ")
    if len(title) > 40:
        title = title[:37] + "..."

    active_lora = lora_path.strip() if (use_lora and lora_path and lora_path.strip()) else None

    params = {
        "captions": captions, "lyrics": lyrics, "bpm": bpm, "key_scale": key_scale,
        "time_signature": time_signature, "vocal_language": vocal_language,
        "inference_steps": inference_steps, "guidance_scale": guidance_scale,
        "random_seed_checkbox": random_seed_checkbox, "seed": seed,
        "reference_audio": reference_audio, "audio_duration": audio_duration,
        "batch_size_input": batch_size_input, "src_audio": src_audio,
        "text2music_audio_code_string": text2music_audio_code_string,
        "repainting_start": repainting_start, "repainting_end": repainting_end,
        "instruction_display_gen": instruction_display_gen,
        "audio_cover_strength": audio_cover_strength,
        "cover_noise_strength": cover_noise_strength, "task_type": task_type,
        "no_fsq": no_fsq, "use_adg": use_adg,
        "cfg_interval_start": cfg_interval_start, "cfg_interval_end": cfg_interval_end,
        "shift": shift, "infer_method": infer_method, "sampler_mode": sampler_mode,
        "velocity_norm_threshold": velocity_norm_threshold,
        "velocity_ema_factor": velocity_ema_factor, "dcw_enabled": dcw_enabled,
        "dcw_mode": dcw_mode, "dcw_scaler": dcw_scaler,
        "dcw_high_scaler": dcw_high_scaler, "dcw_wavelet": dcw_wavelet,
        "custom_timesteps": custom_timesteps, "audio_format": audio_format,
        "mp3_bitrate": mp3_bitrate, "mp3_sample_rate": mp3_sample_rate,
        "lm_temperature": lm_temperature, "think_checkbox": think_checkbox,
        "lm_cfg_scale": lm_cfg_scale, "lm_top_k": lm_top_k, "lm_top_p": lm_top_p,
        "lm_negative_prompt": lm_negative_prompt, "use_cot_metas": use_cot_metas,
        "use_cot_caption": use_cot_caption, "use_cot_language": use_cot_language,
        "is_format_caption": is_format_caption,
        "constrained_decoding_debug": constrained_decoding_debug,
        "allow_lm_batch": allow_lm_batch, "auto_score": auto_score,
        "auto_lrc": auto_lrc, "score_scale": score_scale,
        "lm_batch_chunk_size": lm_batch_chunk_size,
        "enable_normalization": enable_normalization,
        "normalization_db": normalization_db,
        "fade_in_duration": fade_in_duration, "fade_out_duration": fade_out_duration,
        "latent_shift": latent_shift, "latent_rescale": latent_rescale,
        "repaint_mode": repaint_mode, "repaint_strength": repaint_strength,
        "retake_variance": retake_variance, "retake_seed": retake_seed,
    }

    qm = get_task_queue_manager()
    task = qm.add_task(title=title, params=params, lora_path=active_lora, lora_scale=lora_scale)
    gr.Info(t("queue.task_added", title=task.title))
    return refresh_queue_ui_handler()


def refresh_queue_ui_handler() -> tuple[str, list[list[str]], dict[str, Any], str]:
    """Return updated UI representations for the queue status, table, and dropdown."""
    qm = get_task_queue_manager()
    active = qm.get_active_task()

    if active:
        pct = int(active.progress * 100)
        status_md = (
            f"### {t('queue.active_task')}: **{active.title}** (`{active.id}`)\n"
            f"**Status**: {active.status_message} ({pct}%)\n"
            f"**LoRA**: `{active.lora_path or 'Base Model'}`"
        )
    else:
        paused_txt = " *(Paused)*" if qm.is_paused() else ""
        status_md = f"### {t('queue.active_task')}\n*{t('queue.no_active_task')}*{paused_txt}"

    rows = qm.get_table_rows()
    tasks = qm.get_tasks()
    choices = [(f"[{t_obj.id}] {t_obj.title} ({t_obj.status})", t_obj.id) for t_obj in reversed(tasks)]

    btn_label = t("queue.resume_queue_btn") if qm.is_paused() else t("queue.pause_queue_btn")
    return status_md, rows, gr.update(choices=choices), btn_label


def toggle_pause_handler() -> tuple[str, list[list[str]], dict[str, Any], str]:
    """Toggle paused state of queue manager."""
    qm = get_task_queue_manager()
    if qm.is_paused():
        qm.resume()
    else:
        qm.pause()
    return refresh_queue_ui_handler()


def clear_completed_handler() -> tuple[str, list[list[str]], dict[str, Any], str]:
    """Clear completed/failed/cancelled tasks from queue."""
    qm = get_task_queue_manager()
    removed = qm.clear_completed()
    logger.info(f"[TaskQueue] Cleared {removed} completed tasks")
    return refresh_queue_ui_handler()


def _build_task_audio_updates(audio_paths: list[str]) -> tuple[Any, ...]:
    """Return visibility and value updates for the queue's eight audio players."""
    visible_updates = tuple(gr.update(visible=index < len(audio_paths)) for index in range(8))
    audio_values = tuple(audio_paths[index] if index < len(audio_paths) else None for index in range(8))
    return (*visible_updates, *audio_values)


def select_task_handler(selected_task_id: str | None) -> tuple[Any, ...]:
    """Load every generated audio file and the details for a selected task."""
    if not selected_task_id:
        return (*_build_task_audio_updates([]), f"*{t('queue.no_audio')}*")

    qm = get_task_queue_manager()
    task = qm.get_task(selected_task_id)
    if not task:
        return (*_build_task_audio_updates([]), f"*{t('queue.no_audio')}*")
    
    details = (
        f"### Task `{task.id}`: {task.title}\n"
        f"- **Status**: `{task.status}`\n"
        f"- **LoRA Adapter**: `{task.lora_path or 'None (Base Model)'}` (Scale: `{task.lora_scale}`)\n"
        f"- **Duration**: `{task.params.get('audio_duration', 30)}s` | **BPM**: `{task.params.get('bpm', 'Auto')}`\n"
        f"- **Seed**: `{task.params.get('seed') or 'Random'}`\n"
    )
    if task.error_message:
        details += f"\n> Error: **Error**: {task.error_message}\n"
    elif task.generation_info:
        details += f"\n{task.generation_info}\n"

    return (*_build_task_audio_updates(task.output_audio_paths[:8]), details)
