# -*- coding: utf-8 -*-
"""Subprocess worker for isolated task execution with memory cleanup.

Provides the persistent worker subprocess entry point and memory cleanup
utilities.  The worker subprocess loads models independently, processes
tasks received via ``multiprocessing.Queue``, and sends results back.
All CUDA memory allocated inside this process is fully reclaimable when
the process is terminated.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import traceback
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# IPC message types (picklable string constants)
# ---------------------------------------------------------------------------
MSG_INIT_OK = "INIT_OK"
MSG_INIT_FAILED = "INIT_FAILED"
MSG_RESULT = "RESULT"
MSG_HEARTBEAT = "HEARTBEAT"

# Heartbeat interval in seconds — worker sends heartbeats while running a task.
_HEARTBEAT_INTERVAL = 30.0


def cleanup_task_memory() -> None:
    """Release GPU and Python heap memory after a task completes.

    Explicitly runs garbage collection, then clears CUDA caches within this
    process scope only.  Safe to call even when CUDA is unavailable.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def _init_models(
    init_params: dict[str, Any],
) -> tuple[Any, Any, bool]:
    """Load DiT and optional LLM models from serialized init params.

    Args:
        init_params: Picklable dict with all ``initialize_service`` arguments.

    Returns:
        Tuple of ``(dit_handler, llm_handler, success)``.
    """
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    status, ok = dit_handler.initialize_service(
        project_root=init_params["project_root"],
        config_path=init_params["config_path"],
        device=init_params["device"],
        use_flash_attention=init_params.get("use_flash_attention", True),
        compile_model=init_params.get("compile_model", False),
        offload_to_cpu=init_params.get("offload_to_cpu", False),
        offload_dit_to_cpu=init_params.get("offload_dit_to_cpu", False),
        quantization=init_params.get("quantization"),
        use_mlx_dit=init_params.get("use_mlx_dit", True),
        vae_checkpoint=init_params.get("vae_checkpoint"),
    )
    if not ok:
        return dit_handler, llm_handler, False

    if init_params.get("init_llm"):
        checkpoint_dir = os.path.join(init_params["project_root"], "checkpoints")
        lm_status, lm_ok = llm_handler.initialize(
            checkpoint_dir=checkpoint_dir,
            lm_model_path=init_params.get("lm_model_path", ""),
            backend=init_params.get("lm_backend", "pt"),
            device=init_params.get("lm_device", "auto"),
            offload_to_cpu=init_params.get("lm_offload_to_cpu", False),
            dtype=None,
        )
        if not lm_ok:
            logger.warning(f"[SubprocessWorker] LLM init failed: {lm_status}")

    return dit_handler, llm_handler, True


def subprocess_worker_main(
    task_queue: Any,
    result_queue: Any,
    init_params: dict[str, Any],
) -> None:
    """Entry point for the persistent worker subprocess.

    Loads models once, then enters a task loop receiving work from
    *task_queue* and sending results via *result_queue*.

    Args:
        task_queue: ``multiprocessing.Queue`` delivering task tuples or ``None`` (poison pill).
        result_queue: ``multiprocessing.Queue`` for sending results back to the main process.
        init_params: Picklable dict of service-initialization parameters.
    """
    import multiprocessing
    import threading

    try:
        dit_handler, llm_handler, ok = _init_models(init_params)
        if not ok:
            result_queue.put((MSG_INIT_FAILED, "Model initialization failed"))
            return
        result_queue.put((MSG_INIT_OK,))
    except Exception as exc:
        logger.error(f"[SubprocessWorker] Model init exception: {exc}")
        result_queue.put((MSG_INIT_FAILED, str(exc)))
        return

    while True:
        try:
            task_data = task_queue.get(timeout=60)
        except Exception:
            continue

        if task_data is None:
            logger.info("[SubprocessWorker] Received shutdown signal.")
            break

        task_id, title, params, lora_path, lora_scale = task_data

        # Start heartbeat sender in a background thread
        stop_heartbeat = threading.Event()

        def _send_heartbeats() -> None:
            while not stop_heartbeat.is_set():
                try:
                    result_queue.put((MSG_HEARTBEAT, task_id, time.time()))
                except Exception:
                    pass
                stop_heartbeat.wait(_HEARTBEAT_INTERVAL)

        hb_thread = threading.Thread(target=_send_heartbeats, daemon=True)
        hb_thread.start()

        started_at = time.time()
        status = "completed"
        output_paths: list[str] = []
        gen_info = ""
        error_msg: str | None = None

        try:
            from acestep.queue.task_model import GenerationTask
            from acestep.queue.task_worker import apply_task_lora, execute_task

            task = GenerationTask(
                id=task_id, title=title, params=dict(params),
                lora_path=lora_path, lora_scale=lora_scale,
            )
            execute_task(task, dit_handler, llm_handler)

            status = task.status
            output_paths = list(task.output_audio_paths)
            gen_info = task.generation_info
            error_msg = task.error_message
        except Exception as exc:
            status = "failed"
            error_msg = f"{exc}\n{traceback.format_exc()}"
            logger.error(f"[SubprocessWorker] Task {task_id} crashed: {exc}")
        finally:
            stop_heartbeat.set()
            hb_thread.join(timeout=5)
            cleanup_task_memory()

        completed_at = time.time()
        result_queue.put((
            MSG_RESULT, task_id, status, output_paths,
            gen_info, error_msg, started_at, completed_at,
        ))

    cleanup_task_memory()
    logger.info("[SubprocessWorker] Exiting.")
