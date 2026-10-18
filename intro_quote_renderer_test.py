# -*- coding: utf-8 -*-
"""intro_quote_renderer_test.py - Unit tests for intro quote typewriter renderer."""

from __future__ import annotations

import unittest
from intro_quote_renderer import (
    TypewriterConfig,
    build_lines_layout,
    calculate_progress,
    parse_timeline_intro,
    render_frame,
    split_text_lines,
)


class TestIntroQuoteRenderer(unittest.TestCase):
    """Test suite for typewriter parsing, animation math, and frame rendering."""

    def test_parse_timeline_intro_success(self) -> None:
        """Verify timeline.txt parsing splits text1 and the quote remainder accurately."""
        sample_timeline = (
            "Chill Dog on a Sunrise Shore with Lofi Music\n"
            "Hi my friend, welcome the gentle morning light and calm focus.\n"
            "With the new day comes new strength and new thoughts\n"
            "0:00-1:53: Early Song\n"
            "1:53-4:33: Cozy Melody\n"
        )
        title, text1, greeting_rem, quote = parse_timeline_intro(sample_timeline)
        self.assertEqual(title, "Chill Dog on a Sunrise Shore with Lofi Music")
        self.assertEqual(text1, "Hi my friend...")
        self.assertEqual(greeting_rem, "welcome the gentle morning light and calm focus.")
        self.assertEqual(quote, "With the new day comes new strength and new thoughts")

    def test_calculate_progress_in_out(self) -> None:
        """Verify character progress ramps up 0..1 in 2.5s, holds, and ramps down 1..0."""
        # Typewriter In (0 to 2.5s)
        self.assertAlmostEqual(calculate_progress(0.0, 20.0, 2.5), 0.0)
        self.assertAlmostEqual(calculate_progress(1.25, 20.0, 2.5), 0.5)
        self.assertAlmostEqual(calculate_progress(2.5, 20.0, 2.5), 1.0)

        # Hold duration (2.5 to 17.5s)
        self.assertAlmostEqual(calculate_progress(5.0, 20.0, 2.5), 1.0)
        self.assertAlmostEqual(calculate_progress(10.0, 20.0, 2.5), 1.0)
        self.assertAlmostEqual(calculate_progress(17.5, 20.0, 2.5), 1.0)

        # Typewriter Out (17.5 to 20.0s)
        self.assertAlmostEqual(calculate_progress(18.75, 20.0, 2.5), 0.5)
        self.assertAlmostEqual(calculate_progress(20.0, 20.0, 2.5), 0.0)
        self.assertAlmostEqual(calculate_progress(25.0, 20.0, 2.5), 0.0)

    def test_split_text_lines(self) -> None:
        """Verify line wrapping wraps nicely around character limits."""
        text = "welcome the gentle morning light, breathe in the fresh sea breeze, and start your day."
        lines = split_text_lines(text, max_chars=40)
        self.assertGreater(len(lines), 1)
        for line in lines:
            self.assertLessEqual(len(line), 45)

    def test_render_frame_dimensions_and_mode(self) -> None:
        """Verify rendered frame is RGBA with configured dimensions."""
        cfg = TypewriterConfig(width=640, height=360, total_duration=5.0, anim_duration=1.0)
        layout = build_lines_layout("Hi my friend...", "relax today.", "peace", cfg)
        img = render_frame(layout, progress=0.5, config=cfg)
        self.assertEqual(img.size, (640, 360))
        self.assertEqual(img.mode, "RGBA")


if __name__ == "__main__":
    unittest.main()
