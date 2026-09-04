"""Unit tests for startup batch enqueueing helpers."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from acestep.queue.startup_batch import (
    build_default_batch_params,
    enqueue_startup_batch_tasks,
)


class StartupBatchTests(unittest.TestCase):
    """Behavior tests for startup batch parameter builder and queueing."""

    def test_build_default_batch_params_for_turbo_model(self) -> None:
        """Default params for turbo model should use 8 steps and shift 3.0."""
        dit_handler = MagicMock()
        dit_handler.is_turbo_model.return_value = True

        llm_handler = MagicMock()
        llm_handler.llm_initialized = True

        title, params = build_default_batch_params(dit_handler, llm_handler)

        self.assertTrue(len(title) > 0)
        self.assertEqual(8, params["inference_steps"])
        self.assertEqual(3.0, params["shift"])
        self.assertEqual(1.0, params["guidance_scale"])
        self.assertTrue(params["think_checkbox"])
        self.assertEqual(0.3, params["dcw_scaler"])

    def test_build_default_batch_params_for_base_model(self) -> None:
        """Default params for non-turbo model should use 25 steps and 7.0 guidance."""
        dit_handler = MagicMock()
        dit_handler.is_turbo_model.return_value = False

        llm_handler = MagicMock()
        llm_handler.llm_initialized = False

        title, params = build_default_batch_params(dit_handler, llm_handler)

        self.assertEqual(25, params["inference_steps"])
        self.assertEqual(7.0, params["guidance_scale"])
        self.assertFalse(params["think_checkbox"])
        self.assertEqual(0.05, params["dcw_scaler"])

    def test_build_default_batch_params_custom_overrides(self) -> None:
        """Explicit caption, lyrics, and batch_size should override defaults."""
        title, params = build_default_batch_params(
            dit_handler=None,
            llm_handler=None,
            batch_size=4,
            caption="My Custom Acoustic Ballad",
            lyrics="Verse 1\nHello world",
        )

        self.assertEqual("My Custom Acoustic Ballad", title)
        self.assertEqual("My Custom Acoustic Ballad", params["captions"])
        self.assertEqual("Verse 1\nHello world", params["lyrics"])
        self.assertEqual(4, params["batch_size_input"])

    @patch("acestep.queue.startup_batch.get_task_queue_manager")
    def test_enqueue_startup_batch_tasks(self, mock_get_qm: MagicMock) -> None:
        """Enqueueing should initialize handlers and add requested number of tasks."""
        mock_qm = MagicMock()
        mock_qm.add_tasks.return_value = [MagicMock() for _ in range(100)]
        mock_get_qm.return_value = mock_qm

        dit_handler = MagicMock()
        llm_handler = MagicMock()

        tasks = enqueue_startup_batch_tasks(
            count=100,
            dit_handler=dit_handler,
            llm_handler=llm_handler,
        )

        mock_qm.initialize_handlers.assert_called_once_with(dit_handler, llm_handler)
        mock_qm.add_tasks.assert_called_once()
        call_kwargs = mock_qm.add_tasks.call_args.kwargs
        self.assertEqual(100, call_kwargs["count"])
        self.assertEqual(100, len(tasks))


if __name__ == "__main__":
    unittest.main()
