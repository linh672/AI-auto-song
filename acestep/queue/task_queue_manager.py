"""Thread-safe task queue manager with subprocess-isolated execution.

Manages queued generation tasks and dispatches them to a persistent worker
subprocess.  The worker subprocess loads models independently so all CUDA
memory is reclaimable on process termination.  A heartbeat-based timeout
detects stalled tasks and triggers PID-targeted kill + automatic restart.
"""

import multiprocessing
import queue as _queue
import threading
import time
from typing import Any

from loguru import logger

from acestep.queue.task_model import GenerationTask
from acestep.queue.task_subprocess import (
    MSG_HEARTBEAT,
    MSG_INIT_FAILED,
    MSG_INIT_OK,
    MSG_RESULT,
    subprocess_worker_main,
)

# Default per-task heartbeat timeout (seconds).  If no heartbeat is received
# within this window the task is considered stalled.
_DEFAULT_TASK_TIMEOUT = 900  # 15 minutes


class TaskQueueManager:
    """Manages queued generation tasks and background subprocess execution."""

    def __init__(self) -> None:
        self._tasks: list[GenerationTask] = []
        self._lock = threading.Lock()
        self._paused: bool = False
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._dit_handler: Any = None
        self._llm_handler: Any = None
        self._active_task_id: str | None = None
        self._batch_start_time: float | None = None
        self._batch_end_time: float | None = None
        self._last_batch_duration: float | None = None
        self._last_batch_task_count: int = 0
        self._batch_task_count: int = 0
        # Subprocess state
        self._task_timeout_seconds: int = _DEFAULT_TASK_TIMEOUT
        self._init_params: dict[str, Any] | None = None
        self._worker_process: multiprocessing.Process | None = None
        self._task_queue: multiprocessing.Queue | None = None
        self._result_queue: multiprocessing.Queue | None = None

    # ------------------------------------------------------------------
    # Public API: handler registration
    # ------------------------------------------------------------------

    def initialize_handlers(self, dit_handler: Any, llm_handler: Any) -> None:
        """Register model handlers for queue worker.

        The handlers are kept as references for fallback / non-subprocess
        paths (e.g. interactive generation).  The subprocess loads its own
        model instances from ``_init_params``.
        """
        self._dit_handler = dit_handler
        self._llm_handler = llm_handler
        self._ensure_monitor_running()

    def store_init_context(self, init_params: dict[str, Any]) -> None:
        """Store serializable init params for subprocess model loading.

        All values must be picklable primitives (str, int, float, bool, None,
        list, dict of primitives) for safe IPC via ``multiprocessing.Queue``
        on Windows (spawn context).
        """
        self._init_params = init_params

    def set_task_timeout(self, seconds: int) -> None:
        """Set per-task heartbeat timeout in seconds (minimum 60)."""
        self._task_timeout_seconds = max(60, seconds)

    # ------------------------------------------------------------------
    # Public API: task CRUD
    # ------------------------------------------------------------------

    def add_task(
        self, title: str, params: dict[str, Any],
        lora_path: str | None = None, lora_scale: float = 1.0,
    ) -> GenerationTask:
        """Add a new task to the queue."""
        return self.add_tasks(title, params, 1, lora_path, lora_scale)[0]

    def add_tasks(
        self, title: str, params: dict[str, Any], count: int,
        lora_path: str | None = None, lora_scale: float = 1.0,
    ) -> list[GenerationTask]:
        """Atomically add a batch of tasks and start the monitor."""
        tasks = [
            GenerationTask(
                title=title or "Untitled Song", params=dict(params),
                lora_path=lora_path, lora_scale=lora_scale,
            )
            for _ in range(count)
        ]
        with self._lock:
            self._tasks.extend(tasks)
        logger.info(
            f"[TaskQueue] Added {len(tasks)} task(s): '{title}' (LoRA: {lora_path})"
        )
        self._ensure_monitor_running()
        return tasks

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
            self._tasks = [
                t for t in self._tasks if t.status in ("pending", "running")
            ]
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

    def get_timing_info(self) -> dict[str, Any]:
        """Return timing information for the queue and active task."""
        with self._lock:
            now = time.time()
            is_running = self._active_task_id is not None or any(
                t.status == "pending" for t in self._tasks
            )
            batch_elapsed = None
            if self._batch_start_time is not None:
                batch_elapsed = now - self._batch_start_time
            active_elapsed = None
            if self._active_task_id:
                for task in self._tasks:
                    if task.id == self._active_task_id and task.started_at:
                        active_elapsed = now - task.started_at
                        break
            return {
                "is_running": is_running,
                "batch_elapsed": batch_elapsed,
                "active_elapsed": active_elapsed,
                "last_batch_duration": self._last_batch_duration,
                "last_batch_task_count": self._last_batch_task_count,
            }

    def get_table_rows(self) -> list[list[str]]:
        """Return rows for UI dataframe display."""
        with self._lock:
            return [
                [str(index), *task.to_row()]
                for index, task in enumerate(self._tasks, start=1)
            ]

    # ------------------------------------------------------------------
    # Subprocess lifecycle
    # ------------------------------------------------------------------

    def _ensure_monitor_running(self) -> None:
        """Start the monitor thread if not already running."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop, daemon=True,
            )
            self._monitor_thread.start()
            logger.info("[TaskQueue] Monitor thread started")

    def _spawn_worker(self) -> bool:
        """Spawn (or respawn) the worker subprocess.

        Returns True if the worker initialised models successfully.
        """
        if self._init_params is None:
            logger.warning("[TaskQueue] No init_params stored; cannot spawn worker")
            return False

        ctx = multiprocessing.get_context("spawn")
        self._task_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        self._worker_process = ctx.Process(
            target=subprocess_worker_main,
            args=(self._task_queue, self._result_queue, self._init_params),
            daemon=True,
        )
        self._worker_process.start()
        logger.info(
            f"[TaskQueue] Worker subprocess spawned (PID {self._worker_process.pid})"
        )

        # Wait for init result (up to 10 minutes for large model loads)
        try:
            msg = self._result_queue.get(timeout=600)
        except _queue.Empty:
            logger.error("[TaskQueue] Worker init timed out (600s)")
            self._kill_worker()
            return False

        if msg[0] == MSG_INIT_OK:
            logger.info("[TaskQueue] Worker subprocess models loaded successfully")
            return True

        logger.error(f"[TaskQueue] Worker init failed: {msg}")
        self._kill_worker()
        return False

    def _kill_worker(self) -> None:
        """Terminate the worker subprocess by exact PID."""
        proc = self._worker_process
        if proc is None or not proc.is_alive():
            return
        pid = proc.pid
        logger.warning(f"[TaskQueue] Terminating worker subprocess PID {pid}")
        proc.terminate()
        proc.join(timeout=10)
        if proc.is_alive():
            logger.warning(f"[TaskQueue] Force-killing worker PID {pid}")
            proc.kill()
            proc.join(timeout=5)
        self._worker_process = None
        self._task_queue = None
        self._result_queue = None

    def _is_worker_alive(self) -> bool:
        """Check if the worker subprocess is running."""
        return (
            self._worker_process is not None
            and self._worker_process.is_alive()
        )

    # ------------------------------------------------------------------
    # Monitor loop (runs in a daemon thread)
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """Dispatch tasks to the worker subprocess and monitor heartbeats."""
        while not self._stop_event.is_set():
            if self._paused or self._init_params is None:
                time.sleep(1.0)
                continue

            # Find next pending task
            next_task = self._find_next_pending()
            if next_task is None:
                self._finalize_batch_timing()
                self._active_task_id = None
                time.sleep(1.0)
                continue

            # Ensure worker is running
            if not self._is_worker_alive():
                if not self._spawn_worker():
                    logger.error("[TaskQueue] Could not start worker; retrying in 30s")
                    time.sleep(30)
                    continue

            self._start_batch_tracking(next_task)

            # Dispatch task (all values are picklable primitives)
            self._task_queue.put((
                next_task.id,
                next_task.title,
                next_task.params,
                next_task.lora_path,
                next_task.lora_scale,
            ))
            next_task.status = "running"
            next_task.status_message = "Sent to worker subprocess..."

            # Wait for result with heartbeat-based timeout
            timed_out = self._wait_for_result(next_task)

            if timed_out:
                self._handle_task_timeout(next_task)

            with self._lock:
                self._active_task_id = None
            time.sleep(0.5)

    def _find_next_pending(self) -> GenerationTask | None:
        """Return the first pending task, or None."""
        with self._lock:
            for task in self._tasks:
                if task.status == "pending":
                    return task
        return None

    def _start_batch_tracking(self, task: GenerationTask) -> None:
        """Update batch-timing bookkeeping for a new task."""
        with self._lock:
            if self._batch_start_time is None:
                self._batch_start_time = time.time()
                self._batch_task_count = 0
            self._batch_task_count += 1
            self._active_task_id = task.id

    def _finalize_batch_timing(self) -> None:
        """Record batch duration when all tasks are done."""
        with self._lock:
            if self._batch_start_time is not None:
                self._batch_end_time = time.time()
                self._last_batch_duration = (
                    self._batch_end_time - self._batch_start_time
                )
                self._last_batch_task_count = self._batch_task_count
                self._batch_start_time = None

    def _wait_for_result(self, task: GenerationTask) -> bool:
        """Poll the result queue for heartbeats / results.

        The timeout is **heartbeat-based**: the 15-minute window resets on
        every heartbeat received from the worker.  A legitimate long task
        that sends regular heartbeats will never be terminated prematurely.

        Returns True if the task timed out (no heartbeat received within
        ``_task_timeout_seconds``).
        """
        last_heartbeat = time.time()

        while True:
            remaining = self._task_timeout_seconds - (time.time() - last_heartbeat)
            if remaining <= 0:
                return True  # timed out

            try:
                msg = self._result_queue.get(timeout=min(remaining, 5.0))
            except _queue.Empty:
                if not self._is_worker_alive():
                    logger.error(
                        f"[TaskQueue] Worker died while running task {task.id}"
                    )
                    task.status = "failed"
                    task.error_message = "Worker subprocess exited unexpectedly"
                    task.completed_at = time.time()
                    return False
                continue

            if msg[0] == MSG_HEARTBEAT:
                last_heartbeat = time.time()
                continue

            if msg[0] == MSG_RESULT:
                self._apply_result(task, msg)
                return False

            logger.warning(f"[TaskQueue] Unknown message from worker: {msg[0]}")

    def _apply_result(self, task: GenerationTask, msg: tuple) -> None:
        """Update the task object from a worker RESULT message."""
        _, _task_id, status, output_paths, gen_info, error_msg, started, completed = msg
        task.status = status
        task.output_audio_paths = output_paths
        task.generation_info = gen_info
        task.error_message = error_msg
        task.started_at = started
        task.completed_at = completed
        task.progress = 1.0
        task.status_message = (
            "Generation complete!" if status == "completed"
            else f"Error: {error_msg}"
        )
        logger.info(f"[TaskQueue] Task {task.id} finished: {status}")

    def _handle_task_timeout(self, task: GenerationTask) -> None:
        """Handle a timed-out task: kill worker, mark failed, re-enqueue pending."""
        logger.warning(
            f"[TaskQueue] Task {task.id} timed out (no heartbeat for "
            f"{self._task_timeout_seconds}s). Killing worker subprocess."
        )
        task.status = "failed"
        task.error_message = (
            f"Timeout: no heartbeat received for {self._task_timeout_seconds}s"
        )
        task.completed_at = time.time()
        task.progress = 1.0
        task.status_message = f"Timeout after {self._task_timeout_seconds}s"

        self._kill_worker()

        # Count and cancel remaining pending tasks for re-enqueue
        with self._lock:
            pending_tasks = [t for t in self._tasks if t.status == "pending"]
            pending_count = len(pending_tasks)
            for t in pending_tasks:
                t.status = "cancelled"
                t.status_message = "Cancelled for service restart"

        if pending_count > 0:
            self._reenqueue_tasks(pending_count)

    def _reenqueue_tasks(self, count: int) -> None:
        """Re-enqueue *count* tasks using stored batch params after restart."""
        if not self._init_params or count <= 0:
            return

        from acestep.queue.startup_batch import build_default_batch_params

        title, params = build_default_batch_params(
            dit_handler=None,
            llm_handler=None,
            batch_size=self._init_params.get("batch_size"),
            caption=self._init_params.get("batch_caption"),
            lyrics=self._init_params.get("batch_lyrics"),
        )

        self.add_tasks(title=title, params=params, count=count)
        logger.info(
            f"[TaskQueue] Re-enqueued {count} tasks after timeout restart"
        )


_GLOBAL_QUEUE_MANAGER = TaskQueueManager()


def get_task_queue_manager() -> TaskQueueManager:
    """Return the global TaskQueueManager singleton."""
    return _GLOBAL_QUEUE_MANAGER
