# -*- coding: utf-8 -*-
"""Unit tests for the archive_sources functionality in make_video.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from make_video import _safe_move_to_archive, archive_sources, main


class TestArchiveSources(unittest.TestCase):
    """Test suite for archiving used video inputs and gradio_outputs batch directories."""

    def setUp(self) -> None:
        """Create an isolated temporary workspace for each test."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.input_dir = self.root / "input"
        self.gradio_dir = self.root / "gradio_outputs"
        self.archive_dir = self.root / "archive"
        self.output_dir = self.root / "output"

        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.gradio_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.tmp_dir.cleanup()

    def test_archive_success_path(self) -> None:
        """Verify input video and batch folder with sidecars are moved to archive."""
        # 1. Create input video
        video_path = self.input_dir / "abc.mp4"
        video_path.write_bytes(b"dummy video content")

        # 2. Create batch folder with MP3 and sidecars
        batch_folder = self.gradio_dir / "batch_1788987058"
        batch_folder.mkdir(parents=True, exist_ok=True)
        mp3_file = batch_folder / "song.mp3"
        json_file = batch_folder / "song.json"
        session_file = batch_folder / "song.session.npz"

        mp3_file.write_bytes(b"dummy mp3")
        json_file.write_bytes(b"{}")
        session_file.write_bytes(b"session data")

        # 3. Run archive_sources
        archived_video, archived_batches = archive_sources(
            input_video=video_path,
            mp3_files=[mp3_file],
            gradio_outputs_dir=self.gradio_dir,
            archive_dir=self.archive_dir,
        )

        # 4. Assert video moved
        self.assertFalse(video_path.exists())
        self.assertIsNotNone(archived_video)
        self.assertTrue(archived_video.is_file())
        self.assertEqual(archived_video, self.archive_dir / "abc.mp4")

        # 5. Assert batch folder moved
        self.assertFalse(batch_folder.exists())
        self.assertEqual(len(archived_batches), 1)
        dest_batch = self.archive_dir / "batch_1788987058"
        self.assertTrue(dest_batch.is_dir())
        self.assertTrue((dest_batch / "song.mp3").is_file())
        self.assertTrue((dest_batch / "song.json").is_file())
        self.assertTrue((dest_batch / "song.session.npz").is_file())

    def test_deduplicates_multiple_mp3s_in_same_batch(self) -> None:
        """Verify multiple MP3s from the same batch directory are archived once."""
        video_path = self.input_dir / "sample.mp4"
        video_path.write_bytes(b"video")

        batch_folder = self.gradio_dir / "batch_shared"
        batch_folder.mkdir(parents=True, exist_ok=True)
        mp3_1 = batch_folder / "part1.mp3"
        mp3_2 = batch_folder / "part2.mp3"
        mp3_1.write_bytes(b"1")
        mp3_2.write_bytes(b"2")

        archived_video, archived_batches = archive_sources(
            input_video=video_path,
            mp3_files=[mp3_1, mp3_2],
            gradio_outputs_dir=self.gradio_dir,
            archive_dir=self.archive_dir,
        )

        self.assertFalse(batch_folder.exists())
        self.assertEqual(len(archived_batches), 1)
        self.assertTrue((self.archive_dir / "batch_shared" / "part1.mp3").is_file())
        self.assertTrue((self.archive_dir / "batch_shared" / "part2.mp3").is_file())

    def test_file_collision_renames_with_counter(self) -> None:
        """Verify existing file in archive is not overwritten and gets a numeric suffix."""
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        existing_video = self.archive_dir / "clip.mp4"
        existing_video.write_bytes(b"existing version")

        new_video = self.input_dir / "clip.mp4"
        new_video.write_bytes(b"new version")

        moved_path = _safe_move_to_archive(new_video, self.archive_dir)

        self.assertEqual(moved_path.name, "clip_1.mp4")
        self.assertTrue(moved_path.is_file())
        self.assertEqual(moved_path.read_bytes(), b"new version")
        self.assertEqual(existing_video.read_bytes(), b"existing version")

    def test_dir_collision_merges_without_nested_duplication(self) -> None:
        """Verify directory collision merges files into existing dir instead of nesting."""
        existing_batch = self.archive_dir / "batch_100"
        existing_batch.mkdir(parents=True, exist_ok=True)
        (existing_batch / "old_file.txt").write_text("old", encoding="utf-8")

        new_batch = self.gradio_dir / "batch_100"
        new_batch.mkdir(parents=True, exist_ok=True)
        (new_batch / "new_file.txt").write_text("new", encoding="utf-8")

        moved_dir = _safe_move_to_archive(new_batch, self.archive_dir)

        self.assertEqual(moved_dir, existing_batch)
        self.assertFalse(new_batch.exists())
        self.assertFalse((existing_batch / "batch_100").exists())  # No nested directory
        self.assertTrue((existing_batch / "old_file.txt").is_file())
        self.assertTrue((existing_batch / "new_file.txt").is_file())

    def test_archive_handles_missing_files_gracefully(self) -> None:
        """Verify non-existent input video or empty batches do not crash."""
        ghost_video = self.input_dir / "does_not_exist.mp4"
        ghost_mp3 = self.gradio_dir / "batch_ghost" / "ghost.mp3"

        archived_video, archived_batches = archive_sources(
            input_video=ghost_video,
            mp3_files=[ghost_mp3],
            gradio_outputs_dir=self.gradio_dir,
            archive_dir=self.archive_dir,
        )

        self.assertIsNone(archived_video)
        self.assertEqual(archived_batches, [])

    @patch("make_video.archive_sources")
    @patch("make_video._run_ffmpeg_with_progress")
    @patch("make_video._remux_to_ts")
    @patch("make_video._concat_audio")
    @patch("make_video._upscale_video_to_1080p")
    @patch("make_video._enhance_video_with_realesrgan")
    @patch("make_video.strip_container_metadata")
    @patch("make_video._probe_durations_parallel")
    @patch("make_video._find_input_video")
    @patch("make_video._collect_mp3s")
    @patch("make_video._require_ffmpeg")
    def test_main_archive_flag_behavior(
        self,
        mock_ffmpeg: MagicMock,
        mock_collect: MagicMock,
        mock_find_video: MagicMock,
        mock_probe: MagicMock,
        mock_strip: MagicMock,
        mock_enhance: MagicMock,
        mock_upscale: MagicMock,
        mock_concat: MagicMock,
        mock_remux: MagicMock,
        mock_run_ffmpeg: MagicMock,
        mock_archive: MagicMock,
    ) -> None:
        """Verify main() respects the archive_enabled flag."""
        fake_video = self.input_dir / "input.mp4"
        fake_video.write_bytes(b"video")
        fake_mp3 = self.gradio_dir / "batch_1" / "track.mp3"
        fake_mp3.parent.mkdir(parents=True, exist_ok=True)
        fake_mp3.write_bytes(b"mp3")

        mock_collect.return_value = [fake_mp3]
        mock_find_video.return_value = fake_video
        mock_probe.return_value = {fake_video: 10.0, fake_mp3: 10.0}
        mock_strip.return_value = fake_video
        mock_enhance.return_value = fake_video
        mock_upscale.return_value = fake_video
        mock_concat.return_value = (10.0, self.tmp_dir.name, "mp3")

        # Create dummy final output file when ffmpeg completes
        def fake_ffmpeg_run(cmd: list[str], *args: Any, **kwargs: Any) -> None:
            out_file = Path(cmd[-1])
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(b"final video output content")

        mock_run_ffmpeg.side_effect = fake_ffmpeg_run

        with patch("make_video.OUTPUT_DIR", self.output_dir):
            # Test disabled: archive_sources should NOT be called
            main(step_3_enabled=False, archive_enabled=False)
            mock_archive.assert_not_called()

            # Test enabled: archive_sources should be called
            main(step_3_enabled=False, archive_enabled=True)
            self.assertTrue(mock_archive.called)
            args, kwargs = mock_archive.call_args
            self.assertEqual(kwargs.get("input_video"), fake_video)
            self.assertEqual(kwargs.get("mp3_files"), [fake_mp3])


if __name__ == "__main__":
    unittest.main()
