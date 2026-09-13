"""Helper functions for automatically enqueueing batch generation tasks upon service initialization."""

from __future__ import annotations

from typing import Any
import gradio as gr
from loguru import logger

from acestep.ui.gradio.i18n import t


def is_service_init_successful(
    dit_handler: Any,
    enable: bool,
    init_llm: bool,
    lm_success: bool,
    lm_status: str,
) -> bool:
    """Check if service initialization succeeded before enqueueing batch tasks.

    Args:
        dit_handler: Active DiT handler instance.
        enable: Whether DiT initialization enabled generation.
        init_llm: Whether 5Hz LM initialization was requested.
        lm_success: Whether LM initialization returned success.
        lm_status: Status message string returned by LM initialization.

    Returns:
        True if all required service components initialized successfully.
    """
    dit_ready = bool(enable) and (getattr(dit_handler, "model", None) is not None)
    if not dit_ready:
        return False

    if init_llm:
        if not lm_success:
            return False
        if "5Hz LM initialized successfully" not in (lm_status or ""):
            return False

    return True


def maybe_auto_enqueue_batch(
    dit_handler: Any,
    llm_handler: Any,
    count: int = 100,
    batch_size: int | None = None,
    caption: str | None = None,
    lyrics: str | None = None,
) -> list[Any]:
    """Enqueue batch generation tasks into the task queue manager.

    Args:
        dit_handler: Active DiT handler instance.
        llm_handler: Active LLM handler instance.
        count: Number of tasks to enqueue (defaults to 100).
        batch_size: Batch size override for parallel audio generation.
        caption: Optional custom caption text.
        lyrics: Optional custom lyrics text.

    Returns:
        List of created GenerationTask instances.
    """
    from acestep.queue.startup_batch import enqueue_startup_batch_tasks

    tasks = enqueue_startup_batch_tasks(
        count=count,
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        batch_size=batch_size,
        caption=caption,
        lyrics=lyrics,
    )

    logger.info(
        f"[Auto-Batch] Service initialized successfully. Enqueued {len(tasks)} batch tasks."
    )

    try:
        gr.Info(t("queue.tasks_added", count=len(tasks)))
    except Exception:
        pass

    return tasks
