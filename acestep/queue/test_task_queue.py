"""Unit tests for task queue and worker logic."""

import unittest
from unittest.mock import MagicMock, patch

import acestep.ui.gradio.events.results.generation_progress
from acestep.queue.task_model import GenerationTask
from acestep.queue.task_queue_manager import TaskQueueManager
from acestep.queue.task_worker import apply_task_lora, execute_task
from acestep.ui.gradio.events.queue_handlers import select_task_handler


class TestGenerationTask(unittest.TestCase):
    """Test GenerationTask dataclass methods."""

    def test_task_to_row(self):
        task = GenerationTask(
            id="test1234",
            title="Lofi Chill Beat",
            lora_path="/path/to/my_lora",
            status="pending",
        )
        row = task.to_row()
        self.assertEqual(row[0], "test1234")
        self.assertEqual(row[1], "Lofi Chill Beat")
        self.assertEqual(row[2], "my_lora")
        self.assertIn("Pending", row[3])
        self.assertIn("<svg", row[3])


class TestTaskQueueManager(unittest.TestCase):
    """Test TaskQueueManager methods."""

    def setUp(self):
        self.manager = TaskQueueManager()

    def test_add_and_get_task(self):
        task = self.manager.add_task(
            title="Sample Track",
            params={"captions": "A lovely piano song"},
            lora_path="/models/piano_lora",
            lora_scale=0.8,
        )
        self.assertIsNotNone(task.id)
        self.assertEqual(task.title, "Sample Track")
        self.assertEqual(len(self.manager.get_tasks()), 1)
        self.assertEqual(self.manager.get_task(task.id), task)

    def test_get_table_rows_includes_one_based_index(self):
        """Queue table rows include a stable one-based display index."""
        self.manager.add_task(title="First", params={})
        self.manager.add_task(title="Second", params={})

        rows = self.manager.get_table_rows()

        self.assertEqual([row[0] for row in rows], ["1", "2"])
        self.assertEqual([row[2] for row in rows], ["First", "Second"])

    def test_cancel_task(self):
        task = self.manager.add_task(title="Cancel me", params={})
        self.assertEqual(task.status, "pending")
        cancelled = self.manager.cancel_task(task.id)
        self.assertTrue(cancelled)
        self.assertEqual(task.status, "cancelled")

    def test_clear_completed(self):
        t1 = self.manager.add_task(title="Task 1", params={})
        t2 = self.manager.add_task(title="Task 2", params={})
        t1.status = "completed"
        t2.status = "pending"
        
        removed = self.manager.clear_completed()
        self.assertEqual(removed, 1)
        remaining = self.manager.get_tasks()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].id, t2.id)

    def test_pause_and_resume(self):
        self.assertFalse(self.manager.is_paused())

    @patch("acestep.ui.gradio.events.queue_handlers.get_task_queue_manager")
    def test_select_task_returns_all_audio_outputs(self, mock_queue_manager):
        """Selecting a task returns all generated audio paths for display."""
        task = GenerationTask(
            id="preview1",
            title="Preview Track",
            output_audio_paths=["/outputs/first.mp3", "/outputs/second.mp3"],
        )
        self.manager._tasks.append(task)
        mock_queue_manager.return_value = self.manager

        updates = select_task_handler(task.id)

        self.assertEqual(len(updates), 17)
        self.assertEqual(updates[8:10], ("/outputs/first.mp3", "/outputs/second.mp3"))
        self.assertIn("Preview Track", updates[-1])
        self.manager.pause()
        self.assertTrue(self.manager.is_paused())
        self.manager.resume()
        self.assertFalse(self.manager.is_paused())


class TestTaskWorker(unittest.TestCase):
    """Test task execution and LoRA switching."""

    def test_apply_task_lora_load(self):
        dit_mock = MagicMock()
        dit_mock.model = MagicMock()
        dit_mock.lora_loaded = False

        apply_task_lora(dit_mock, "/models/jazz_lora", lora_scale=0.7)
        dit_mock.load_lora.assert_called_once_with("/models/jazz_lora")
        dit_mock.set_lora_scale.assert_called_once_with(0.7)

    def test_apply_task_lora_unload_when_none(self):
        dit_mock = MagicMock()
        dit_mock.model = MagicMock()
        dit_mock.lora_loaded = True

        apply_task_lora(dit_mock, None)
        dit_mock.unload_lora.assert_called_once()

    @patch("acestep.ui.gradio.events.results.generation_progress.generate_with_progress")
    def test_execute_task_success(self, mock_gen):
        dit_mock = MagicMock()
        dit_mock.model = MagicMock()
        llm_mock = MagicMock()

        # Mock generator returning output
        mock_gen.return_value = [
            (None, None, None, None, None, None, None, None, ["/path/audio.mp3"], "info string")
        ]

        task = GenerationTask(
            id="exec1",
            title="Execute Track",
            params={"captions": "Rock anthem"},
            lora_path="/models/rock_lora",
        )

        execute_task(task, dit_mock, llm_mock)
        self.assertEqual(task.status, "completed")
        self.assertEqual(task.output_audio_paths, ["/path/audio.mp3"])
        self.assertEqual(task.progress, 1.0)


if __name__ == "__main__":
    unittest.main()
