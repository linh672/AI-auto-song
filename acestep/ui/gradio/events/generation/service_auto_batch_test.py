"""Unit tests for service_auto_batch helper functions and auto-batch on init flow."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from acestep.ui.gradio.events.generation.service_auto_batch import (
    is_service_init_successful,
    maybe_auto_enqueue_batch,
)
from acestep.ui.gradio.events.generation.service_init import init_service_wrapper


class TestIsServiceInitSuccessful(unittest.TestCase):
    """Test validation of service initialization success conditions."""

    def test_returns_true_when_dit_and_llm_succeed(self) -> None:
        """Both DiT and 5Hz LM successfully initialized with expected success status."""
        dit_handler = MagicMock()
        dit_handler.model = MagicMock()
        lm_status = (
            "✅ 5Hz LM initialized successfully\n"
            "Model: C:\\checkpoints\\acestep-5Hz-lm-1.7B\n"
            "Device: NVIDIA GeForce RTX 4060 Laptop GPU\n"
            "GPU Memory Utilization: 0.720\n"
            "Low GPU Memory Mode: True"
        )
        self.assertTrue(
            is_service_init_successful(
                dit_handler=dit_handler,
                enable=True,
                init_llm=True,
                lm_success=True,
                lm_status=lm_status,
            )
        )

    def test_returns_false_when_lm_status_lacks_success_message(self) -> None:
        """LM status without '5Hz LM initialized successfully' must fail validation."""
        dit_handler = MagicMock()
        dit_handler.model = MagicMock()
        lm_status = "❌ Failed to load LM model: CUDA out of memory"
        self.assertFalse(
            is_service_init_successful(
                dit_handler=dit_handler,
                enable=True,
                init_llm=True,
                lm_success=True,
                lm_status=lm_status,
            )
        )

    def test_returns_false_when_lm_success_is_false(self) -> None:
        """lm_success=False must fail validation even if status string contains keywords."""
        dit_handler = MagicMock()
        dit_handler.model = MagicMock()
        self.assertFalse(
            is_service_init_successful(
                dit_handler=dit_handler,
                enable=True,
                init_llm=True,
                lm_success=False,
                lm_status="5Hz LM initialized successfully",
            )
        )

    def test_returns_false_when_dit_enable_is_false(self) -> None:
        """DiT enable flag False must fail validation."""
        dit_handler = MagicMock()
        dit_handler.model = MagicMock()
        self.assertFalse(
            is_service_init_successful(
                dit_handler=dit_handler,
                enable=False,
                init_llm=True,
                lm_success=True,
                lm_status="✅ 5Hz LM initialized successfully",
            )
        )

    def test_returns_false_when_dit_model_is_none(self) -> None:
        """dit_handler.model being None must fail validation."""
        dit_handler = MagicMock()
        dit_handler.model = None
        self.assertFalse(
            is_service_init_successful(
                dit_handler=dit_handler,
                enable=True,
                init_llm=True,
                lm_success=True,
                lm_status="✅ 5Hz LM initialized successfully",
            )
        )

    def test_returns_true_when_init_llm_false_and_dit_succeeds(self) -> None:
        """When init_llm=False, only DiT success is required."""
        dit_handler = MagicMock()
        dit_handler.model = MagicMock()
        self.assertTrue(
            is_service_init_successful(
                dit_handler=dit_handler,
                enable=True,
                init_llm=False,
                lm_success=False,
                lm_status="",
            )
        )


class TestMaybeAutoEnqueueBatch(unittest.TestCase):
    """Test batch enqueueing helper."""

    @patch("acestep.queue.startup_batch.enqueue_startup_batch_tasks")
    def test_enqueues_100_tasks_by_default(self, mock_enqueue: MagicMock) -> None:
        """maybe_auto_enqueue_batch should call enqueue_startup_batch_tasks with count=100."""
        mock_enqueue.return_value = [MagicMock() for _ in range(100)]
        dit_handler = MagicMock()
        llm_handler = MagicMock()

        tasks = maybe_auto_enqueue_batch(dit_handler, llm_handler)

        mock_enqueue.assert_called_once_with(
            count=100,
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            batch_size=None,
            caption=None,
            lyrics=None,
        )
        self.assertEqual(100, len(tasks))

    @patch("acestep.queue.startup_batch.enqueue_startup_batch_tasks")
    def test_forwards_overrides(self, mock_enqueue: MagicMock) -> None:
        """Custom count, batch_size, caption, and lyrics should be passed along."""
        mock_enqueue.return_value = [MagicMock() for _ in range(100)]
        dit_handler = MagicMock()
        llm_handler = MagicMock()

        tasks = maybe_auto_enqueue_batch(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            count=100,
            batch_size=2,
            caption="Pop synthwave track",
            lyrics="[Verse]\nNeon nights",
        )

        mock_enqueue.assert_called_once_with(
            count=100,
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            batch_size=2,
            caption="Pop synthwave track",
            lyrics="[Verse]\nNeon nights",
        )
        self.assertEqual(100, len(tasks))


class TestInitServiceWrapperAutoBatchIntegration(unittest.TestCase):
    """Test auto_add_batch parameter in init_service_wrapper."""

    @patch("acestep.ui.gradio.events.generation.service_init.get_global_gpu_config")
    @patch("acestep.ui.gradio.events.generation.service_init.maybe_auto_enqueue_batch")
    def test_auto_batch_enqueues_when_enabled_and_init_succeeds(
        self,
        mock_auto_enqueue: MagicMock,
        mock_gpu_config: MagicMock,
    ) -> None:
        """When auto_add_batch=True and service initializes successfully, 100 tasks are enqueued."""
        mock_gpu_config.return_value = MagicMock(
            available_lm_models=["acestep-5Hz-lm-1.7B"],
            lm_backend_restriction=None,
            tier="tier6",
            gpu_memory_gb=24.0,
            max_duration_with_lm=600,
            max_duration_without_lm=600,
            max_batch_size_with_lm=4,
            max_batch_size_without_lm=8,
        )
        mock_auto_enqueue.return_value = [MagicMock() for _ in range(100)]

        dit_handler = MagicMock()
        dit_handler.initialize_service.return_value = ("DiT loaded successfully", True)
        dit_handler.model = MagicMock()
        dit_handler.is_turbo_model.return_value = True

        llm_handler = MagicMock()
        llm_handler.llm_initialized = True
        llm_handler.initialize.return_value = (
            "✅ 5Hz LM initialized successfully\nModel: acestep-5Hz-lm-1.7B",
            True,
        )

        result = init_service_wrapper(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            checkpoint="/checkpoints",
            config_path="acestep-v15-turbo",
            device="cuda",
            init_llm=True,
            lm_model_path="acestep-5Hz-lm-1.7B",
            backend="vllm",
            use_flash_attention=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
            compile_model=False,
            quantization=False,
            auto_add_batch=True,
            captions="Upbeat Disco",
            lyrics="",
        )

        mock_auto_enqueue.assert_called_once_with(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            count=100,
            batch_size=2,
            caption="Upbeat Disco",
            lyrics="",
        )
        status = result[0]
        self.assertIn("Auto-Batch: Enqueued 100 tasks into generation queue", status)

    @patch("acestep.ui.gradio.events.generation.service_init.get_global_gpu_config")
    @patch("acestep.ui.gradio.events.generation.service_init.maybe_auto_enqueue_batch")
    def test_auto_batch_skipped_when_lm_fails(
        self,
        mock_auto_enqueue: MagicMock,
        mock_gpu_config: MagicMock,
    ) -> None:
        """When LM initialization fails, auto-batch must NOT enqueue tasks."""
        mock_gpu_config.return_value = MagicMock(
            available_lm_models=["acestep-5Hz-lm-1.7B"],
            lm_backend_restriction=None,
            tier="tier6",
            gpu_memory_gb=24.0,
            max_duration_with_lm=600,
            max_duration_without_lm=600,
            max_batch_size_with_lm=4,
            max_batch_size_without_lm=8,
        )

        dit_handler = MagicMock()
        dit_handler.initialize_service.return_value = ("DiT loaded successfully", True)
        dit_handler.model = MagicMock()
        dit_handler.is_turbo_model.return_value = True

        llm_handler = MagicMock()
        llm_handler.llm_initialized = False
        llm_handler.initialize.return_value = (
            "❌ 5Hz LM initialization failed: OOM",
            False,
        )

        result = init_service_wrapper(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            checkpoint="/checkpoints",
            config_path="acestep-v15-turbo",
            device="cuda",
            init_llm=True,
            lm_model_path="acestep-5Hz-lm-1.7B",
            backend="vllm",
            use_flash_attention=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
            compile_model=False,
            quantization=False,
            auto_add_batch=True,
        )

        mock_auto_enqueue.assert_not_called()
        status = result[0]
        self.assertNotIn("Auto-Batch: Enqueued", status)

    @patch("acestep.ui.gradio.events.generation.service_init.get_global_gpu_config")
    @patch("acestep.ui.gradio.events.generation.service_init.maybe_auto_enqueue_batch")
    def test_auto_batch_skipped_when_disabled(
        self,
        mock_auto_enqueue: MagicMock,
        mock_gpu_config: MagicMock,
    ) -> None:
        """When auto_add_batch=False, no tasks are enqueued even on successful init."""
        mock_gpu_config.return_value = MagicMock(
            available_lm_models=["acestep-5Hz-lm-1.7B"],
            lm_backend_restriction=None,
            tier="tier6",
            gpu_memory_gb=24.0,
            max_duration_with_lm=600,
            max_duration_without_lm=600,
            max_batch_size_with_lm=4,
            max_batch_size_without_lm=8,
        )

        dit_handler = MagicMock()
        dit_handler.initialize_service.return_value = ("DiT loaded successfully", True)
        dit_handler.model = MagicMock()
        dit_handler.is_turbo_model.return_value = True

        llm_handler = MagicMock()
        llm_handler.llm_initialized = True
        llm_handler.initialize.return_value = (
            "✅ 5Hz LM initialized successfully\nModel: acestep-5Hz-lm-1.7B",
            True,
        )

        init_service_wrapper(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            checkpoint="/checkpoints",
            config_path="acestep-v15-turbo",
            device="cuda",
            init_llm=True,
            lm_model_path="acestep-5Hz-lm-1.7B",
            backend="vllm",
            use_flash_attention=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
            compile_model=False,
            quantization=False,
            auto_add_batch=False,
        )

        mock_auto_enqueue.assert_not_called()


if __name__ == "__main__":
    unittest.main()
