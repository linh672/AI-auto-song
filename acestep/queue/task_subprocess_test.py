# -*- coding: utf-8 -*-
"""Unit tests for subprocess worker and memory cleanup utilities."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

from acestep.queue.task_subprocess import (
    MSG_HEARTBEAT,
    MSG_INIT_FAILED,
    MSG_INIT_OK,
    MSG_RESULT,
    _init_models,
    cleanup_task_memory,
)


class TestCleanupTaskMemory(unittest.TestCase):
    """Test memory cleanup utility behavior."""

    @patch("gc.collect")
    def test_cleanup_calls_gc_and_cuda_empty_cache(self, mock_gc: MagicMock) -> None:
        """Verify gc.collect and torch.cuda caches are cleared when CUDA is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict(sys.modules, {"torch": mock_torch}):
            cleanup_task_memory()

        mock_gc.assert_called_once()
        mock_torch.cuda.synchronize.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called_once()
        mock_torch.cuda.ipc_collect.assert_called_once()

    @patch("gc.collect")
    def test_cleanup_handles_no_cuda(self, mock_gc: MagicMock) -> None:
        """Verify cleanup does not call cuda methods when CUDA is unavailable."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            cleanup_task_memory()

        mock_gc.assert_called_once()
        mock_torch.cuda.empty_cache.assert_not_called()
        mock_torch.cuda.ipc_collect.assert_not_called()

    @patch("gc.collect")
    def test_cleanup_handles_import_error(self, mock_gc: MagicMock) -> None:
        """Verify cleanup succeeds even if torch cannot be imported."""
        with patch.dict(sys.modules, {"torch": None}):
            cleanup_task_memory()

        mock_gc.assert_called_once()


class TestInitModels(unittest.TestCase):
    """Test model initialization in the worker subprocess."""

    @patch("acestep.handler.AceStepHandler")
    @patch("acestep.llm_inference.LLMHandler")
    def test_init_models_success_dit_only(
        self,
        mock_llm_cls: MagicMock,
        mock_dit_cls: MagicMock,
    ) -> None:
        """Verify DiT-only initialization succeeds."""
        mock_dit = MagicMock()
        mock_dit.initialize_service.return_value = ("Ready", True)
        mock_dit_cls.return_value = mock_dit

        init_params = {
            "project_root": "/project",
            "config_path": "/project/config.json",
            "device": "cuda:0",
            "init_llm": False,
        }

        dit_handler, llm_handler, ok = _init_models(init_params)

        self.assertTrue(ok)
        mock_dit.initialize_service.assert_called_once_with(
            project_root="/project",
            config_path="/project/config.json",
            device="cuda:0",
            use_flash_attention=True,
            compile_model=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
            quantization=None,
            use_mlx_dit=True,
            vae_checkpoint=None,
        )

    @patch("acestep.handler.AceStepHandler")
    @patch("acestep.llm_inference.LLMHandler")
    def test_init_models_failure(
        self,
        mock_llm_cls: MagicMock,
        mock_dit_cls: MagicMock,
    ) -> None:
        """Verify failure in DiT initialization is propagated."""
        mock_dit = MagicMock()
        mock_dit.initialize_service.return_value = ("OOM", False)
        mock_dit_cls.return_value = mock_dit

        init_params = {
            "project_root": "/project",
            "config_path": "/project/config.json",
            "device": "cuda:0",
        }

        _, _, ok = _init_models(init_params)
        self.assertFalse(ok)


class TestIPCConstants(unittest.TestCase):
    """Test IPC message constants."""

    def test_constants_are_strings(self) -> None:
        """Verify IPC constants are string primitives."""
        self.assertEqual(MSG_INIT_OK, "INIT_OK")
        self.assertEqual(MSG_INIT_FAILED, "INIT_FAILED")
        self.assertEqual(MSG_RESULT, "RESULT")
        self.assertEqual(MSG_HEARTBEAT, "HEARTBEAT")


if __name__ == "__main__":
    unittest.main()
