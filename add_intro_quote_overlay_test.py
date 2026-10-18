# -*- coding: utf-8 -*-
"""add_intro_quote_overlay_test.py - Unit tests for add_intro_quote_overlay CLI orchestrator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from add_intro_quote_overlay import find_target_videos


class TestAddIntroQuoteOverlay(unittest.TestCase):
    """Test suite for target video discovery and argument parsing."""

    def test_find_target_videos_with_mock_archive(self) -> None:
        """Verify discovery finds folders with timeline.txt and matching video files."""
        with tempfile.TemporaryDirectory() as tmp_root:
            archive = Path(tmp_root)
            batch = archive / "test_batch_01"
            batch.mkdir()

            timeline = batch / "timeline.txt"
            timeline.write_text(
                "Chill Dog on a Sunset Beach with Lofi Music\n"
                "Hi my friend, unwind and relax.\n"
                "Peace comes from within\n"
                "0:00-2:16: Song One\n",
                encoding="utf-8",
            )

            # Create mock video file
            mock_video = batch / "Chill Dog on a Sunset Beach with Lofi Music.mp4"
            mock_video.write_bytes(b"\x00" * 1024)

            targets = find_target_videos(archive)
            self.assertEqual(len(targets), 1)
            batch_dir, vid_path, title, t1, gr, q = targets[0]
            self.assertEqual(batch_dir, batch)
            self.assertEqual(vid_path, mock_video)
            self.assertEqual(title, "Chill Dog on a Sunset Beach with Lofi Music")
            self.assertEqual(t1, "Hi my friend...")
            self.assertEqual(gr, "unwind and relax.")
            self.assertEqual(q, "Peace comes from within")


if __name__ == "__main__":
    unittest.main()
