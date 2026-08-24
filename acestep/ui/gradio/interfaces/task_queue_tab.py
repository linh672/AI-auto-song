"""Task Queue tab interface for managing background generation jobs."""

from typing import Any
import gradio as gr

from acestep.ui.gradio.i18n import t


def create_task_queue_section() -> dict[str, Any]:
    """Create the Task Queue interface tab.

    Returns:
        Dictionary of Gradio components for the task queue tab.
    """
    gr.HTML(
        """
        <div style="text-align: center; padding: 10px; margin-bottom: 15px;">
            <h2>Generation Task Queue</h2>
            <p>Queue generation tasks with specific LoRAs and process them sequentially in the background</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            queue_status_box = gr.Markdown(
                value=f"### {t('queue.active_task')}\n*{t('queue.no_active_task')}*",
                elem_classes=["has-info-container"],
            )
        with gr.Column(scale=2):
            with gr.Row():
                toggle_pause_btn = gr.Button(
                    t("queue.pause_queue_btn"),
                    variant="secondary",
                    size="sm",
                )
                clear_completed_btn = gr.Button(
                    t("queue.clear_completed_btn"),
                    variant="secondary",
                    size="sm",
                )
                refresh_queue_btn = gr.Button(
                    t("queue.refresh_btn"),
                    variant="primary",
                    size="sm",
                )

    with gr.Row():
        queue_table = gr.Dataframe(
            headers=[
                t("queue.col_id"),
                t("queue.col_title"),
                t("queue.col_lora"),
                t("queue.col_status"),
                t("queue.col_created"),
            ],
            datatype=["str", "str", "str", "str", "str"],
            value=[],
            interactive=False,
            elem_id="task-queue-table",
        )

    with gr.Accordion(t("queue.audio_preview"), open=True):
        with gr.Row():
            with gr.Column(scale=2):
                task_select_dropdown = gr.Dropdown(
                    label=t("queue.select_task"),
                    choices=[],
                    value=None,
                    interactive=True,
                )
                task_audio_preview = gr.Audio(
                    label=t("queue.audio_preview"),
                    type="filepath",
                    interactive=False,
                )
            with gr.Column(scale=3):
                task_details_markdown = gr.Markdown(
                    value=f"*{t('queue.no_audio')}*"
                )

    return {
        "queue_status_box": queue_status_box,
        "toggle_pause_btn": toggle_pause_btn,
        "clear_completed_btn": clear_completed_btn,
        "refresh_queue_btn": refresh_queue_btn,
        "queue_table": queue_table,
        "task_select_dropdown": task_select_dropdown,
        "task_audio_preview": task_audio_preview,
        "task_details_markdown": task_details_markdown,
    }
