# -*- coding: utf-8 -*-
"""intro_quote_renderer.py - Renders aesthetic typewriter intro text frames for chill lofi videos."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageDraw, ImageFont


@dataclass
class TypewriterConfig:
    """Configuration for typewriter overlay animation and layout."""

    total_duration: float = 20.0
    anim_duration: float = 2.5
    fps: int = 24
    width: int = 1920
    height: int = 1080
    font_family: str = "segoepr.ttf"
    quote_font: str = "georgiai.ttf"


def parse_timeline_intro(content: str) -> tuple[str, str, str, str]:
    """Parse title, text1, greeting remainder, and quote from timeline.txt content.

    Args:
        content: Raw text content of timeline.txt.

    Returns:
        Tuple of (title, text1, text2_greeting, text2_quote).
    """
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    title = lines[0] if lines else "Chill Dog with Lofi Music"
    line2 = lines[1] if len(lines) > 1 else "Hi my friend, relax and enjoy the moment."
    line3 = lines[2] if len(lines) > 2 and not lines[2].startswith("0:00") else ""

    text1 = "Hi my friend..."
    greeting_rem = line2
    lower_line2 = line2.lower()
    for prefix in ("hi my friend,", "hi my friend...", "hi my friend.", "hi my friend:"):
        if lower_line2.startswith(prefix):
            greeting_rem = line2[len(prefix):].strip()
            break

    return title, text1, greeting_rem, line3


def calculate_progress(time_sec: float, total_sec: float = 20.0, anim_sec: float = 2.5) -> float:
    """Calculate character visibility progress (0.0 to 1.0) with in/out typewriter animation."""
    if time_sec < 0.0 or time_sec >= total_sec:
        return 0.0
    if time_sec <= anim_sec:
        return max(0.0, min(1.0, time_sec / anim_sec))
    if time_sec < (total_sec - anim_sec):
        return 1.0
    return max(0.0, min(1.0, (total_sec - time_sec) / anim_sec))


def find_system_font(font_name: str) -> str:
    """Find absolute path to a system font file, falling back gracefully."""
    windir = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"
    for candidate in (font_name, "segoepr.ttf", "Candara.ttf", "georgia.ttf", "arial.ttf"):
        p = windir / candidate
        if p.exists():
            return str(p)
    return font_name


def split_text_lines(text: str, max_chars: int = 65) -> list[str]:
    """Wrap a long text line into multiple shorter lines around word boundaries."""
    words = text.split()
    if not words:
        return []
    lines = []
    current: list[str] = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 > max_chars and current:
            lines.append(" ".join(current))
            current = [w]
            cur_len = len(w)
        else:
            current.append(w)
            cur_len += len(w) + 1
    if current:
        lines.append(" ".join(current))
    return lines


def build_lines_layout(
    text1: str,
    greeting: str,
    quote: str,
    config: TypewriterConfig,
) -> list[tuple[str, ImageFont.FreeTypeFont, tuple[int, int, int], int]]:
    """Build prepared line list with assigned fonts, colors, and line spacing."""
    title_font = ImageFont.truetype(find_system_font(config.font_family), 46)
    body_font = ImageFont.truetype(find_system_font(config.font_family), 30)
    quote_font = ImageFont.truetype(find_system_font(config.quote_font), 32)

    layout: list[tuple[str, ImageFont.FreeTypeFont, tuple[int, int, int], int]] = []
    layout.append((text1, title_font, (255, 255, 255), 18))

    for line in split_text_lines(greeting, max_chars=68):
        layout.append((line, body_font, (240, 240, 245), 8))

    if quote:
        quote_text = f'"{quote.strip()}"'
        for qline in split_text_lines(quote_text, max_chars=65):
            layout.append((qline, quote_font, (255, 228, 160), 8))

    return layout


def render_frame(
    lines_layout: list[tuple[str, ImageFont.FreeTypeFont, tuple[int, int, int], int]],
    progress: float,
    config: TypewriterConfig,
) -> Image.Image:
    """Render a single 1920x1080 RGBA transparent frame with typewriter text overlay."""
    img = Image.new("RGBA", (config.width, config.height), (0, 0, 0, 0))
    if progress <= 0.0:
        return img

    draw = ImageDraw.Draw(img)
    total_chars = sum(len(text) for text, _, _, _ in lines_layout)
    chars_to_show = int(round(total_chars * progress))

    # Calculate total height of text block
    total_h = 0
    line_bboxes = []
    for text, font, _, spacing in lines_layout:
        bbox = draw.textbbox((0, 0), text, font=font)
        h = bbox[3] - bbox[1]
        line_bboxes.append((bbox[2] - bbox[0], h))
        total_h += h + spacing

    card_padding_x, card_padding_y = 60, 36
    max_w = max(w for w, _ in line_bboxes)
    card_w = min(config.width - 160, max_w + card_padding_x * 2)
    card_h = total_h + card_padding_y * 2
    card_x = (config.width - card_w) // 2
    card_y = 110

    # Smooth fade for card background
    card_alpha = int(115 * min(1.0, progress * 2.5))
    draw.rounded_rectangle(
        [card_x, card_y, card_x + card_w, card_y + card_h],
        radius=22,
        fill=(0, 0, 0, card_alpha),
        outline=(255, 255, 255, int(35 * min(1.0, progress * 2.5))),
        width=1,
    )

    remaining_chars = chars_to_show
    cur_y = card_y + card_padding_y
    for (text, font, color, spacing), (_, h) in zip(lines_layout, line_bboxes):
        if remaining_chars <= 0:
            break
        visible_text = text[:remaining_chars]
        remaining_chars -= len(text)

        sub_bbox = draw.textbbox((0, 0), visible_text, font=font)
        sub_w = sub_bbox[2] - sub_bbox[0]
        sub_x = (config.width - sub_w) // 2

        draw.text(
            (sub_x, cur_y),
            visible_text,
            font=font,
            fill=(*color, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 180),
        )
        cur_y += h + spacing

    return img


def stream_typewriter_rawvideo(
    lines_layout: list[tuple[str, ImageFont.FreeTypeFont, tuple[int, int, int], int]],
    config: TypewriterConfig,
    out_stream: BinaryIO,
) -> None:
    """Stream all raw RGBA frames sequentially to a binary pipe for ffmpeg."""
    import gc
    total_frames = int(config.total_duration * config.fps)
    for f in range(total_frames):
        prog = calculate_progress(f / config.fps, config.total_duration, config.anim_duration)
        frame = render_frame(lines_layout, prog, config)
        out_stream.write(frame.tobytes())
        del frame
    out_stream.flush()
    gc.collect()
