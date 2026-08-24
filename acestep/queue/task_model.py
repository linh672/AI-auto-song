"""Dataclass definitions for the generation task queue."""

from dataclasses import dataclass, field
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

        status_icons = {
            "pending": "⏳ Pending",
            "running": "🔄 Running",
            "completed": "✅ Done",
            "failed": "❌ Failed",
            "cancelled": "🚫 Cancelled",
        }
        status_display = status_icons.get(self.status, self.status)

        return [
            self.id,
            self.title[:30] + ("..." if len(self.title) > 30 else ""),
            lora_display,
            status_display,
            created_str,
        ]
