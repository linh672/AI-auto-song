# -*- coding: utf-8 -*-
"""Unit tests for batch_video.py batch orchestration logic."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

from batch_video import (
    _archive_batch_files,
    _list_input_videos,
    _take_batch,
    batch_main,
)


class TestListInputVideos(unittest.TestCase):
    """Tests for _list_input_videos helper."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.tmp.name) / "input"
        self.input_dir.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_returns_sorted_mp4s(self) -> None:
        """MP4 files are returned in alphabetical order."""
        (self.input_dir / "z_video.mp4").write_bytes(b"v")
        (self.input_dir / "a_video.mp4").write_bytes(b"v")
        result = _list_input_videos(self.input_dir)
        self.assertEqual([p.name for p in result], ["a_video.mp4", "z_video.mp4"])

    def test_empty_directory_returns_empty(self) -> None:
        """Empty input directory yields an empty list (no exception)."""
        result = _list_input_videos(self.input_dir)
        self.assertEqual(result, [])

    def test_ignores_non_mp4_files(self) -> None:
        """Non-MP4 files are excluded."""
        (self.input_dir / "readme.txt").write_bytes(b"x")
        (self.input_dir / "clip.mp4").write_bytes(b"v")
        result = _list_input_videos(self.input_dir)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "clip.mp4")


class TestTakeBatch(unittest.TestCase):
    """Tests for _take_batch slice-from-pool helper."""

    def test_takes_exact_count(self) -> None:
        """Takes exactly *size* items and removes them from the pool."""
        pool = [Path(f"{i}.mp3") for i in range(10)]
        batch = _take_batch(pool, 3)
        self.assertEqual(len(batch), 3)
        self.assertEqual(len(pool), 7)
        self.assertEqual(batch[0], Path("0.mp3"))

    def test_pool_order_preserved(self) -> None:
        """Remaining pool items stay in original order after take."""
        pool = [Path(f"{i}.mp3") for i in range(5)]
        _take_batch(pool, 2)
        self.assertEqual(pool[0], Path("2.mp3"))


class TestArchiveBatchFiles(unittest.TestCase):
    """Tests for _archive_batch_files safe-move logic."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.batch_dir = self.root / "archive" / "test_batch"
        self.batch_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_moves_video_and_audio(self) -> None:
        """Input video and audio files are moved into the batch archive."""
        video = self.root / "input" / "clip.mp4"
        video.parent.mkdir()
        video.write_bytes(b"video_data")

        audio_dir = self.root / "gradio_outputs" / "batch_001"
        audio_dir.mkdir(parents=True)
        audio_files = []
        for i in range(3):
            f = audio_dir / f"track_{i}.mp3"
            f.write_bytes(b"audio")
            audio_files.append(f)

        _archive_batch_files(video, audio_files, self.batch_dir)

        # Video moved to batch root
        self.assertTrue((self.batch_dir / "clip.mp4").exists())
        self.assertFalse(video.exists())

        # Audio moved to used_audio/
        used_audio_dir = self.batch_dir / "used_audio"
        self.assertTrue(used_audio_dir.is_dir())
        self.assertEqual(len(list(used_audio_dir.iterdir())), 3)

    def test_cleans_empty_parent_dirs(self) -> None:
        """Empty parent directories of moved audio files are cleaned up."""
        audio_dir = self.root / "gradio_outputs" / "empty_batch"
        audio_dir.mkdir(parents=True)
        f = audio_dir / "track.mp3"
        f.write_bytes(b"audio")

        video = self.root / "v.mp4"
        video.write_bytes(b"v")

        _archive_batch_files(video, [f], self.batch_dir)

        # The now-empty batch folder should be removed
        self.assertFalse(audio_dir.exists())


class TestBatchMain(unittest.TestCase):
    """Integration-level tests for batch_main orchestration."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.input_dir = self.root / "input"
        self.gradio_dir = self.root / "gradio_outputs"
        self.archive_dir = self.root / "archive"
        self.input_dir.mkdir(parents=True)
        self.gradio_dir.mkdir(parents=True)
        self.archive_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _create_videos(self, count: int) -> list[Path]:
        """Helper: create *count* dummy MP4 files in input_dir."""
        videos = []
        for i in range(count):
            v = self.input_dir / f"video_{i:02d}.mp4"
            v.write_bytes(b"video_content")
            videos.append(v)
        return videos

    def _create_audio(self, count: int) -> list[Path]:
        """Helper: create *count* dummy MP3 files in gradio_dir."""
        files = []
        for i in range(count):
            batch_dir = self.gradio_dir / f"batch_{i // 10:03d}"
            batch_dir.mkdir(exist_ok=True)
            f = batch_dir / f"track_{i:04d}.mp3"
            f.write_bytes(b"audio_content")
            files.append(f)
        return files

    @patch("batch_video.main")
    def test_processes_all_videos_with_sufficient_audio(
        self, mock_main: MagicMock
    ) -> None:
        """Two videos with 250 audio files -> 2 batches of 5, loop completes."""
        batch_size = 5
        self._create_videos(2)
        self._create_audio(12)

        batch_main(
            batch_size=batch_size,
            step_3_enabled=False,
            input_dir=self.input_dir,
            gradio_outputs_dir=self.gradio_dir,
            archive_dir=self.archive_dir,
        )

        self.assertEqual(mock_main.call_count, 2)
        # Each call should receive exactly batch_size audio files
        for c in mock_main.call_args_list:
            self.assertEqual(len(c.kwargs["mp3_files"]), batch_size)
            self.assertFalse(c.kwargs["archive_enabled"])

    @patch("batch_video.main")
    def test_stops_when_insufficient_audio(
        self, mock_main: MagicMock
    ) -> None:
        """With 3 audio files and batch_size=5, no batch should be processed."""
        self._create_videos(1)
        self._create_audio(3)

        batch_main(
            batch_size=5,
            step_3_enabled=False,
            input_dir=self.input_dir,
            gradio_outputs_dir=self.gradio_dir,
            archive_dir=self.archive_dir,
        )

        mock_main.assert_not_called()

    @patch("batch_video.main")
    def test_archive_structure_correct(
        self, mock_main: MagicMock
    ) -> None:
        """Archive folder contains video, final mp4 (mocked), and used_audio/."""
        batch_size = 3
        self._create_videos(1)
        self._create_audio(5)

        batch_main(
            batch_size=batch_size,
            step_3_enabled=False,
            input_dir=self.input_dir,
            gradio_outputs_dir=self.gradio_dir,
            archive_dir=self.archive_dir,
        )

        # One batch folder should exist under archive
        batch_dirs = [d for d in self.archive_dir.iterdir() if d.is_dir()]
        self.assertEqual(len(batch_dirs), 1)

        batch_dir = batch_dirs[0]
        # Input video should be moved into the batch dir
        self.assertTrue((batch_dir / "video_00.mp4").exists())
        # used_audio/ should contain batch_size files
        used_audio = batch_dir / "used_audio"
        self.assertTrue(used_audio.is_dir())
        audio_files = list(used_audio.rglob("*.mp3"))
        self.assertEqual(len(audio_files), batch_size)

    @patch("batch_video.main")
    def test_failed_batch_preserves_files(
        self, mock_main: MagicMock
    ) -> None:
        """When main() raises, source files are NOT moved to archive."""
        batch_size = 3
        videos = self._create_videos(1)
        audio = self._create_audio(5)

        mock_main.side_effect = RuntimeError("ffmpeg exploded")

        batch_main(
            batch_size=batch_size,
            step_3_enabled=False,
            input_dir=self.input_dir,
            gradio_outputs_dir=self.gradio_dir,
            archive_dir=self.archive_dir,
        )

        # Video should still be in input/
        self.assertTrue(videos[0].exists())
        # All audio should still exist in gradio_outputs
        remaining = list(self.gradio_dir.rglob("*.mp3"))
        self.assertEqual(len(remaining), 5)

    @patch("batch_video.main")
    def test_no_input_videos_exits_cleanly(
        self, mock_main: MagicMock
    ) -> None:
        """No input videos -> no processing, clean exit."""
        self._create_audio(10)

        batch_main(
            batch_size=5,
            input_dir=self.input_dir,
            gradio_outputs_dir=self.gradio_dir,
            archive_dir=self.archive_dir,
        )

        mock_main.assert_not_called()

    @patch("batch_video.main")
    def test_partial_batch_stops_at_boundary(
        self, mock_main: MagicMock
    ) -> None:
        """3 videos but only 10 audio with batch_size=5 -> only 2 batches run."""
        self._create_videos(3)
        self._create_audio(10)

        batch_main(
            batch_size=5,
            step_3_enabled=False,
            input_dir=self.input_dir,
            gradio_outputs_dir=self.gradio_dir,
            archive_dir=self.archive_dir,
        )

        self.assertEqual(mock_main.call_count, 2)


if __name__ == "__main__":
    unittest.main()
