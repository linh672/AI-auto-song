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

    def test_add_tasks_adds_a_batch_atomically(self):
        """Bulk queue insertion creates independent tasks in their original order."""
        tasks = self.manager.add_tasks("Batch", {"captions": "test"}, 3)

        self.assertEqual(len(tasks), 3)
        self.assertEqual(self.manager.get_tasks(), tasks)
        self.assertEqual(len({task.id for task in tasks}), 3)
        self.assertIsNot(tasks[0].params, tasks[1].params)

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
        self.manager.pause()
        self.assertTrue(self.manager.is_paused())
        self.manager.resume()
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

    @patch("acestep.ui.gradio.events.queue_handlers.get_task_queue_manager")
    @patch("gradio.Info")
    def test_add_to_queue_handler_batch(self, mock_info, mock_qm):
        """add_to_queue_handler enqueues multiple tasks when queue_count > 1."""
        from acestep.ui.gradio.events.queue_handlers import add_to_queue_handler

        mock_qm.return_value = self.manager
        kwargs = {
            "captions": "A batch song",
            "lyrics": "",
            "bpm": None,
            "key_scale": "",
            "time_signature": "",
            "vocal_language": "en",
            "inference_steps": 25,
            "guidance_scale": 7.0,
            "random_seed_checkbox": True,
            "seed": "",
            "reference_audio": None,
            "audio_duration": 30.0,
            "batch_size_input": 1,
            "src_audio": None,
            "text2music_audio_code_string": "",
            "repainting_start": 0.0,
            "repainting_end": 0.0,
            "instruction_display_gen": "",
            "audio_cover_strength": 0.5,
            "cover_noise_strength": 0.0,
            "task_type": "text2music",
            "no_fsq": False,
            "use_adg": False,
            "cfg_interval_start": 0.0,
            "cfg_interval_end": 1.0,
            "shift": 1.0,
            "infer_method": "ode",
            "sampler_mode": "euler",
            "velocity_norm_threshold": 0.0,
            "velocity_ema_factor": 0.0,
            "dcw_enabled": False,
            "dcw_mode": "double",
            "dcw_scaler": 0.05,
            "dcw_high_scaler": 0.02,
            "dcw_wavelet": "haar",
            "custom_timesteps": "",
            "audio_format": "mp3",
            "mp3_bitrate": "320k",
            "mp3_sample_rate": 48000,
            "lm_temperature": 0.85,
            "think_checkbox": False,
            "lm_cfg_scale": 2.0,
            "lm_top_k": 0,
            "lm_top_p": 0.9,
            "lm_negative_prompt": "NO USER INPUT",
            "use_cot_metas": True,
            "use_cot_caption": False,
            "use_cot_language": True,
            "is_format_caption": False,
            "constrained_decoding_debug": False,
            "allow_lm_batch": True,
            "auto_score": False,
            "auto_lrc": False,
            "score_scale": 0.1,
            "lm_batch_chunk_size": 8,
            "enable_normalization": False,
            "normalization_db": -14.0,
            "fade_in_duration": 0.0,
            "fade_out_duration": 0.0,
            "latent_shift": 0.0,
            "latent_rescale": 1.0,
            "repaint_mode": "balanced",
            "repaint_strength": 0.5,
            "retake_variance": 0.0,
            "retake_seed": "",
            "lora_path": "",
            "use_lora": False,
            "lora_scale": 1.0,
            "queue_count": 3,
        }
        add_to_queue_handler(**kwargs)
        self.assertEqual(len(self.manager.get_tasks()), 3)
        mock_info.assert_called_once()


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

    @patch("acestep.ui.gradio.events.results.generation_progress.generate_with_progress")
    def test_execute_task_marks_generation_error_as_failed(self, mock_gen):
        """Generation errors do not appear as successful zero-audio tasks."""
        mock_gen.return_value = [
            (None,) * 8 + (None, "generation info", "❌ VAE decoder failed", None)
        ]
        task = GenerationTask(id="failed1", title="Failed Track", params={})

        execute_task(task, MagicMock(), MagicMock())

        self.assertEqual(task.status, "failed")
        self.assertEqual(task.output_audio_paths, [])
        self.assertEqual(task.error_message, "❌ VAE decoder failed")
        self.assertIsNotNone(task.started_at)
        self.assertIsNotNone(task.completed_at)


class TestQueueTiming(unittest.TestCase):
    """Test timing tracking and duration formatting."""

    def test_format_duration(self):
        from acestep.ui.gradio.events.queue_handlers import _format_duration

        self.assertEqual(_format_duration(45), "0m 45s")
        self.assertEqual(_format_duration(125), "2m 05s")
        self.assertEqual(_format_duration(3665), "1h 01m 05s")

    def test_queue_timing_info(self):
        manager = TaskQueueManager()
        timing = manager.get_timing_info()
        self.assertFalse(timing["is_running"])
        self.assertIsNone(timing["batch_elapsed"])

        manager._batch_start_time = 100.0
        with patch("time.time", return_value=150.0):
            timing = manager.get_timing_info()
            self.assertEqual(timing["batch_elapsed"], 50.0)

    @patch("acestep.ui.gradio.events.queue_handlers.get_task_queue_manager")
    def test_refresh_queue_ui_shows_last_batch_duration(self, mock_qm):
        from acestep.ui.gradio.events.queue_handlers import refresh_queue_ui_handler

        manager = TaskQueueManager()
        manager._last_batch_duration = 75.0
        manager._last_batch_task_count = 3
        mock_qm.return_value = manager

        status_md, _, _, _ = refresh_queue_ui_handler()
        self.assertIn("Total Queue Duration", status_md)
        self.assertIn("1m 15s", status_md)
        self.assertIn("3 tasks completed", status_md)


if __name__ == "__main__":
    unittest.main()
