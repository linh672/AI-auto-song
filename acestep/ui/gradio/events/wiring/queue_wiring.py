"""Wiring for task queue events."""

from typing import Any
from .. import queue_handlers as q_handlers
from acestep.queue.task_queue_manager import get_task_queue_manager


def register_queue_handlers(
    generation_section: dict[str, Any],
    results_section: dict[str, Any],
    queue_section: dict[str, Any],
    dit_handler: Any,
    llm_handler: Any,
) -> None:
    """Wire up all Task Queue events and initialize worker handlers."""
    # Initialize singleton handlers
    get_task_queue_manager().initialize_handlers(dit_handler, llm_handler)

    queue_ui_outputs = [
        queue_section["queue_status_box"],
        queue_section["queue_table"],
        queue_section["task_select_dropdown"],
        queue_section["toggle_pause_btn"],
    ]

    # Add to Queue click
    if "add_to_queue_btn" in generation_section:
        generation_section["add_to_queue_btn"].click(
            fn=q_handlers.add_to_queue_handler,
            inputs=[
                generation_section["captions"],
                generation_section["lyrics"],
                generation_section["bpm"],
                generation_section["key_scale"],
                generation_section["time_signature"],
                generation_section["vocal_language"],
                generation_section["inference_steps"],
                generation_section["guidance_scale"],
                generation_section["random_seed_checkbox"],
                generation_section["seed"],
                generation_section["reference_audio"],
                generation_section["audio_duration"],
                generation_section["batch_size_input"],
                generation_section["src_audio"],
                generation_section["text2music_audio_code_string"],
                generation_section["repainting_start"],
                generation_section["repainting_end"],
                generation_section["instruction_display_gen"],
                generation_section["audio_cover_strength"],
                generation_section["cover_noise_strength"],
                generation_section["task_type"],
                generation_section["no_fsq"],
                generation_section["use_adg"],
                generation_section["cfg_interval_start"],
                generation_section["cfg_interval_end"],
                generation_section["shift"],
                generation_section["infer_method"],
                generation_section["sampler_mode"],
                generation_section["velocity_norm_threshold"],
                generation_section["velocity_ema_factor"],
                generation_section["dcw_enabled"],
                generation_section["dcw_mode"],
                generation_section["dcw_scaler"],
                generation_section["dcw_high_scaler"],
                generation_section["dcw_wavelet"],
                generation_section["custom_timesteps"],
                generation_section["audio_format"],
                generation_section["mp3_bitrate"],
                generation_section["mp3_sample_rate"],
                generation_section["lm_temperature"],
                generation_section["think_checkbox"],
                generation_section["lm_cfg_scale"],
                generation_section["lm_top_k"],
                generation_section["lm_top_p"],
                generation_section["lm_negative_prompt"],
                generation_section["use_cot_metas"],
                generation_section["use_cot_caption"],
                generation_section["use_cot_language"],
                results_section["is_format_caption_state"],
                generation_section["constrained_decoding_debug"],
                generation_section["allow_lm_batch"],
                generation_section["auto_score"],
                generation_section["auto_lrc"],
                generation_section["score_scale"],
                generation_section["lm_batch_chunk_size"],
                generation_section["enable_normalization"],
                generation_section["normalization_db"],
                generation_section["fade_in_duration"],
                generation_section["fade_out_duration"],
                generation_section["latent_shift"],
                generation_section["latent_rescale"],
                generation_section["repaint_mode"],
                generation_section["repaint_strength"],
                generation_section["retake_variance"],
                generation_section["retake_seed"],
                generation_section["lora_path"],
                generation_section["use_lora_checkbox"],
                generation_section["lora_scale_slider"],
            ],
            outputs=queue_ui_outputs,
        )

    # Refresh queue table
    queue_section["refresh_queue_btn"].click(
        fn=q_handlers.refresh_queue_ui_handler,
        outputs=queue_ui_outputs,
    )

    # Toggle pause
    queue_section["toggle_pause_btn"].click(
        fn=q_handlers.toggle_pause_handler,
        outputs=queue_ui_outputs,
    )

    # Clear completed
    queue_section["clear_completed_btn"].click(
        fn=q_handlers.clear_completed_handler,
        outputs=queue_ui_outputs,
    )

    # Select task dropdown
    queue_section["task_select_dropdown"].change(
        fn=q_handlers.select_task_handler,
        inputs=[queue_section["task_select_dropdown"]],
        outputs=[
            *queue_section["task_audio_columns"],
            *queue_section["task_audio_previews"],
            queue_section["task_details_markdown"],
        ],
    )
