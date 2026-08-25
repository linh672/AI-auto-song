"""Thread-safe task queue manager and singleton."""

import threading
import time
from typing import Any
from loguru import logger

from acestep.queue.task_model import GenerationTask
from acestep.queue.task_worker import execute_task


class TaskQueueManager:
    """Manages queued generation tasks and background execution."""

    def __init__(self) -> None:
        self._tasks: list[GenerationTask] = []
        self._lock = threading.Lock()
        self._paused: bool = False
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._dit_handler: Any = None
        self._llm_handler: Any = None
        self._active_task_id: str | None = None

    def initialize_handlers(self, dit_handler: Any, llm_handler: Any) -> None:
        """Register model handlers for queue worker."""
        self._dit_handler = dit_handler
        self._llm_handler = llm_handler
        self._ensure_worker_running()

    def _ensure_worker_running(self) -> None:
        """Start worker thread if not already running."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()
            logger.info("[TaskQueue] Worker thread started")

    def add_task(
        self,
        title: str,
        params: dict[str, Any],
        lora_path: str | None = None,
        lora_scale: float = 1.0,
    ) -> GenerationTask:
        """Add a new task to the queue."""
        task = GenerationTask(
            title=title or "Untitled Song",
            params=params,
            lora_path=lora_path,
            lora_scale=lora_scale,
        )
        with self._lock:
            self._tasks.append(task)
        logger.info(f"[TaskQueue] Added task {task.id}: '{task.title}' (LoRA: {task.lora_path})")
        self._ensure_worker_running()
        return task

    def get_tasks(self) -> list[GenerationTask]:
        """Return shallow copy of all tasks."""
        with self._lock:
            return list(self._tasks)

    def get_task(self, task_id: str) -> GenerationTask | None:
        """Find a task by its ID."""
        with self._lock:
            for task in self._tasks:
                if task.id == task_id:
                    return task
        return None

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        with self._lock:
            for task in self._tasks:
                if task.id == task_id and task.status == "pending":
                    task.status = "cancelled"
                    task.status_message = "Cancelled by user"
                    return True
        return False

    def delete_task(self, task_id: str) -> bool:
        """Remove a task from the list if not running."""
        with self._lock:
            for i, task in enumerate(self._tasks):
                if task.id == task_id and task.status != "running":
                    self._tasks.pop(i)
                    return True
        return False

    def clear_completed(self) -> int:
        """Remove all completed, failed, or cancelled tasks."""
        with self._lock:
            before = len(self._tasks)
            self._tasks = [t for t in self._tasks if t.status in ("pending", "running")]
            return before - len(self._tasks)

    def pause(self) -> None:
        """Pause the queue worker."""
        self._paused = True
        logger.info("[TaskQueue] Queue paused")

    def resume(self) -> None:
        """Resume the queue worker."""
        self._paused = False
        logger.info("[TaskQueue] Queue resumed")

    def is_paused(self) -> bool:
        """Check if the queue is paused."""
        return self._paused

    def get_active_task(self) -> GenerationTask | None:
        """Get the currently running task."""
        if not self._active_task_id:
            return None
        return self.get_task(self._active_task_id)

    def get_table_rows(self) -> list[list[str]]:
        """Return rows for UI dataframe display."""
        with self._lock:
            return [
                [str(index), *task.to_row()]
                for index, task in enumerate(self._tasks, start=1)
            ]

    def _worker_loop(self) -> None:
        """Background thread loop to process tasks sequentially."""
        while not self._stop_event.is_set():
            if self._paused or self._dit_handler is None:
                time.sleep(1.0)
                continue

            next_task = None
            with self._lock:
                for task in self._tasks:
                    if task.status == "pending":
                        next_task = task
                        break

            if next_task is None:
                self._active_task_id = None
                time.sleep(1.0)
                continue

            self._active_task_id = next_task.id
            logger.info(f"[TaskQueue] Starting execution of task {next_task.id}: {next_task.title}")
            execute_task(next_task, self._dit_handler, self._llm_handler)
            self._active_task_id = None
            time.sleep(0.5)


_GLOBAL_QUEUE_MANAGER = TaskQueueManager()


def get_task_queue_manager() -> TaskQueueManager:
    """Return the global TaskQueueManager singleton."""
    return _GLOBAL_QUEUE_MANAGER
