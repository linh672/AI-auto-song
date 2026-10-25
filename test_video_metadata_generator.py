# -*- coding: utf-8 -*-
"""Unit tests for video_metadata_generator.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from video_metadata_generator import (
    VideoMetadata,
    _resolve_agy_binary,
    extract_json_payload,
    extract_video_frame,
    generate_metadata_via_agy,
    generate_offline_metadata,
    generate_video_metadata,
    sanitize_filename,
)


class TestSanitizeFilename(unittest.TestCase):
    """Tests for sanitize_filename."""

    def test_removes_illegal_characters(self) -> None:
        raw = 'Chill Dog: Sunset Beach / Ocean? * " < > |'
        sanitized = sanitize_filename(raw)
        self.assertEqual(sanitized, "Chill Dog Sunset Beach  Ocean")

    def test_fallback_when_empty(self) -> None:
        self.assertEqual(sanitize_filename(":::***???"), "Chill_Dog_Lofi_Video")


class TestExtractJsonPayload(unittest.TestCase):
    """Tests for extract_json_payload with ANSI codes and CLI output."""

    def test_clean_json(self) -> None:
        raw = '{"title": "Chill Dog", "greeting": "Hi", "quote": "Peace", "song_titles": ["Track 1"]}'
        result = extract_json_payload(raw)
        self.assertEqual(result["title"], "Chill Dog")

    def test_json_with_ansi_codes_and_logs(self) -> None:
        raw = (
            "\x1b[32m[INFO] Starting agy...\x1b[0m\n"
            "Warning: some non-fatal CLI notice\n"
            '{"title": "Sunset Shore", "greeting": "Hello", "quote": "Be calm", "song_titles": ["A", "B"]}\n'
            "\x1b[34mDone in 1.2s\x1b[0m"
        )
        result = extract_json_payload(raw)
        self.assertEqual(result["title"], "Sunset Shore")
        self.assertEqual(result["song_titles"], ["A", "B"])

    def test_json_wrapped_in_markdown(self) -> None:
        raw = "Here is your metadata:\n```json\n{\"title\": \"Zen Dog\", \"greeting\": \"Hi\", \"quote\": \"Q\", \"song_titles\": []}\n```"
        result = extract_json_payload(raw)
        self.assertEqual(result["title"], "Zen Dog")

    def test_unwraps_agy_response_envelope(self) -> None:
        raw = (
            '{"conversation_id": "abc-123", "status": "SUCCESS", '
            '"response": "```json\\n{\\"title\\": \\"Chill Sunset\\", \\"greeting\\": \\"Hi\\", \\"quote\\": \\"Q\\", \\"song_titles\\": [\\"T1\\"]}\\n```"}'
        )
        result = extract_json_payload(raw)
        self.assertEqual(result["title"], "Chill Sunset")
        self.assertEqual(result["song_titles"], ["T1"])

    def test_missing_json_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            extract_json_payload("No json here at all!")


class TestExtractVideoFrame(unittest.TestCase):
    """Tests for extract_video_frame FFmpeg command."""

    @patch("subprocess.run")
    def test_uses_5s_for_long_video(self, mock_run: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video_p = Path(tmp) / "input.mp4"
            out_p = Path(tmp) / "preview.jpg"
            extract_video_frame(video_p, out_p, duration=30.0)

            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            self.assertIn("-ss", cmd)
            idx = cmd.index("-ss")
            self.assertEqual(cmd[idx + 1], "5.00")

    @patch("subprocess.run")
    def test_uses_midpoint_for_short_video(self, mock_run: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video_p = Path(tmp) / "input.mp4"
            out_p = Path(tmp) / "preview.jpg"
            extract_video_frame(video_p, out_p, duration=6.0)

            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            idx = cmd.index("-ss")
            self.assertEqual(cmd[idx + 1], "3.00")


class TestGenerateOfflineMetadata(unittest.TestCase):
    """Tests for offline vocabulary fallback."""

    def test_generates_exact_count_of_unique_song_titles(self) -> None:
        p = Path("cristofer-beach-scene.jpg")
        meta = generate_offline_metadata(p, song_count=25)
        self.assertIn("Chill Vibes", meta.title)
        self.assertNotIn("Dog", meta.title)
        self.assertTrue(meta.greeting.startswith("Hi my friend"))
        self.assertTrue(len(meta.quote) > 5)
        self.assertEqual(len(meta.song_titles), 25)
        self.assertEqual(len(set(meta.song_titles)), 25)


class TestGenerateMetadataViaAgy(unittest.TestCase):
    """Tests for agy CLI orchestration."""

    @patch("video_metadata_generator._resolve_agy_binary", return_value=Path("C:/bin/agy.exe"))
    @patch("subprocess.run")
    def test_passes_image_flag_and_parses_result(self, mock_run: MagicMock, _mock_bin: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            stdout='{"title": "Beach Lofi", "greeting": "Hi my friend", "quote": "Calm", "song_titles": ["T1", "T2"]}',
            returncode=0,
        )
        img_p = Path("preview.jpg")
        meta = generate_metadata_via_agy(img_p, song_count=2)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        prompt_arg = cmd[cmd.index("-p") + 1]
        self.assertIn(str(img_p.resolve()), prompt_arg)
        self.assertIn("--dangerously-skip-permissions", cmd)
        self.assertIn("--model", cmd)
        self.assertEqual(cmd[cmd.index("--model") + 1], "gemini-3.8-flash-high")
        self.assertIn("--effort", cmd)
        self.assertEqual(cmd[cmd.index("--effort") + 1], "high")
        self.assertIn("--output-format", cmd)
        self.assertEqual(meta.title, "Beach Lofi")
        self.assertEqual(meta.song_titles, ["T1", "T2"])


class TestGenerateVideoMetadataOrchestrator(unittest.TestCase):
    """Tests for fallback when agy fails."""

    @patch("video_metadata_generator.generate_metadata_via_agy", side_effect=RuntimeError("CLI error"))
    def test_falls_back_gracefully_on_error(self, _mock_agy: MagicMock) -> None:
        meta = generate_video_metadata(Path("sunset-shore.jpg"), song_count=5)
        self.assertIsInstance(meta, VideoMetadata)
        self.assertEqual(len(meta.song_titles), 5)


if __name__ == "__main__":
    unittest.main()
