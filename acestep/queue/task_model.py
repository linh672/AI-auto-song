"""Dataclass definitions for the generation task queue."""

from dataclasses import dataclass, field
from html import escape
import time
from typing import Any
import uuid


@dataclass
class GenerationTask:
    """Represents a queued music generation task."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = "Untitled Song"
    created_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending, running, completed, failed, cancelled
    progress: float = 0.0
    status_message: str = "Pending in queue"

    # LoRA configuration for this specific task
    lora_path: str | None = None
    lora_scale: float = 1.0

    # Generation parameters dictionary
    params: dict[str, Any] = field(default_factory=dict)

    # Outputs
    output_audio_paths: list[str] = field(default_factory=list)
    generation_info: str = ""
    error_message: str | None = None

    def to_row(self) -> list[str]:
        """Convert task to a row representation for UI dataframes."""
        created_str = time.strftime("%H:%M:%S", time.localtime(self.created_at))
        lora_display = (
            self.lora_path.split("/")[-1].split("\\")[-1]
            if self.lora_path
            else "None (Base)"
        )

        status_display = {
            "pending": ("Pending", "#eab308", "M12 6v6l4 2"),
            "running": ("Running", "#3b82f6", "M12 6v6l4 2"),
            "completed": ("Done", "#22c55e", "m5 12 4 4L19 6"),
            "failed": ("Failed", "#ef4444", "m6 6 12 12M18 6 6 18"),
            "cancelled": ("Cancelled", "#94a3b8", "M6 6l12 12M18 6 6 18"),
        }
        status_label, status_color, status_path = status_display.get(
            self.status,
            (self.status, "#94a3b8", "M12 8v4m0 4h.01"),
        )
        status_html = (
            '<span style="align-items:center;display:inline-flex;gap:8px">'
            f'<svg aria-hidden="true" fill="none" height="20" '
            f'stroke="{status_color}" stroke-linecap="round" stroke-linejoin="round" '
            'stroke-width="2.5" viewBox="0 0 24 24" width="20">'
            f'<circle cx="12" cy="12" r="9"></circle><path d="{status_path}"></path></svg>'
            f"<span>{escape(status_label)}</span></span>"
        )

        return [
            self.id,
            self.title[:30] + ("..." if len(self.title) > 30 else ""),
            lora_display,
            status_html,
            created_str,
        ]
