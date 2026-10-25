# -*- coding: utf-8 -*-
"""video_metadata_generator.py - Generates video titling, quotes, and song names via agy or fallback."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

_FALLBACK_ADJECTIVES = ["Kindred", "Cozy", "Peaceful", "Calm", "Dreamy", "Velvet", "Gleaming", "Golden", "Solar", "Mellow"]
_FALLBACK_NOUNS = ["Shallow", "Foliage", "Lagoon", "Glade", "Trance", "Petals", "Afterglow", "Whispers", "Echoes", "Haven"]
_FALLBACK_QUOTES = [
    "Peace comes from within. Do not seek it without",
    "With the new day comes new strength and new thoughts",
    "In the middle of difficulty lies opportunity",
    "Silence is a source of great strength",
]

DEFAULT_AGY_MODEL = os.environ.get("ACE_STEP_AGY_MODEL", "gemini-3.8-flash-high")
DEFAULT_AGY_EFFORT = os.environ.get("ACE_STEP_AGY_EFFORT", "high")
DEFAULT_AGY_TIMEOUT = int(os.environ.get("ACE_STEP_AGY_TIMEOUT", "300"))


@dataclass
class VideoMetadata:
    """Container for video metadata generated for titles, intros, and playlists."""

    title: str
    greeting: str
    quote: str
    song_titles: list[str]


def sanitize_filename(name: str) -> str:
    """Sanitize title string into safe Windows filename."""
    cleaned = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    return cleaned or "Chill_Dog_Lofi_Video"


def _resolve_agy_binary() -> Path | None:
    """Resolve agy binary path from PATH or user AppData directory."""
    which_path = shutil.which("agy")
    if which_path:
        return Path(which_path)
    fallback = Path(os.environ.get("LOCALAPPDATA", "")) / "agy" / "bin" / "agy.exe"
    return fallback if fallback.is_file() else None


def extract_json_payload(raw_output: str) -> dict:
    """Extract and parse JSON object between the first '{' and last '}'."""
    clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", raw_output).strip()
    start = clean.find("{")
    end = clean.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No valid JSON object found in output:\n{raw_output}")
    parsed = json.loads(clean[start : end + 1])
    if isinstance(parsed, dict) and "response" in parsed and "title" not in parsed:
        return extract_json_payload(str(parsed["response"]))
    return parsed


def extract_video_frame(video_path: Path, out_path: Path, duration: float = 0.0) -> Path:
    """Extract a representative preview image at 5.0s (or midpoint if short)."""
    extract_time = min(5.0, duration / 2.0) if duration > 0.0 else 5.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{extract_time:.2f}",
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception as exc:
        logger.warning(f"Frame extraction failed for {video_path.name} ({exc}).")
    return out_path


def generate_offline_metadata(frame_path: Path, song_count: int) -> VideoMetadata:
    """Generate high-quality default metadata using filename keywords and vocabulary banks."""
    stem = frame_path.stem.replace("_", " ").replace("-", " ")
    scene_words = [w.capitalize() for w in stem.split() if len(w) > 3 and w.lower() not in {"unsplash", "batch", "preview"}]
    scene_desc = " ".join(scene_words[:2]) if scene_words else "Cozy Shore"

    title = f"Chill Vibes on a {scene_desc} with Lofi Music"
    greeting = f"Hi my friend, relax and enjoy peaceful moments by the {scene_desc.lower()}."
    quote = _FALLBACK_QUOTES[hash(stem) % len(_FALLBACK_QUOTES)]

    song_titles: list[str] = []
    used: set[str] = set()
    idx = 0
    while len(song_titles) < song_count:
        adj = _FALLBACK_ADJECTIVES[idx % len(_FALLBACK_ADJECTIVES)]
        noun = _FALLBACK_NOUNS[(idx // len(_FALLBACK_ADJECTIVES)) % len(_FALLBACK_NOUNS)]
        suffix = f" Pt. {idx // (len(_FALLBACK_ADJECTIVES) * len(_FALLBACK_NOUNS)) + 1}" if idx >= 100 else ""
        candidate = f"{adj} {noun}{suffix}"
        if candidate not in used:
            used.add(candidate)
            song_titles.append(candidate)
        idx += 1

    return VideoMetadata(title=title, greeting=greeting, quote=quote, song_titles=song_titles)


def generate_metadata_via_agy(
    preview_img: Path,
    song_count: int,
    model: str = DEFAULT_AGY_MODEL,
    effort: str = DEFAULT_AGY_EFFORT,
    timeout: int = DEFAULT_AGY_TIMEOUT,
) -> VideoMetadata:
    """Call Antigravity CLI to analyze preview image and return structured metadata."""
    agy_bin = _resolve_agy_binary()
    if not agy_bin:
        raise FileNotFoundError("Antigravity CLI (agy) binary not found.")

    rel_img = preview_img.resolve()
    prompt = (
        f"Analyze the scene in the image @{rel_img}. "
        "1. Give me a title for this video that names the ACTUAL main subject you see "
        "(person, animal, or scenery), e.g. 'Chill Girl on a Sunset Beach with Lofi Music'. "
        "2. Write a warm greeting starting with 'Hi my friend, ...'. "
        "3. Provide a short inspirational chill quote. "
        f"4. Generate exactly {song_count} unique, creative chill/lofi song titles. "
        "Return ONLY a valid JSON object matching this schema: "
        '{"title": "...", "greeting": "...", "quote": "...", "song_titles": ["...", ...]}'
    )
    cmd = [
        str(agy_bin),
        "--dangerously-skip-permissions",
        "--model", model,
        "--effort", effort,
        "-p", prompt,
        "--output-format", "text",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
    data = extract_json_payload(result.stdout)
    return VideoMetadata(
        title=str(data["title"]),
        greeting=str(data["greeting"]),
        quote=str(data["quote"]),
        song_titles=[str(t) for t in data["song_titles"]],
    )


def generate_video_metadata(frame_path: Path, song_count: int, video_duration: float = 0.0) -> VideoMetadata:
    """Attempt agy vision generation with automatic graceful offline fallback."""
    try:
        logger.info(f"Generating video metadata via Antigravity CLI for {frame_path.name}...")
        meta = generate_metadata_via_agy(frame_path, song_count)
        if meta.title and len(meta.song_titles) >= song_count:
            return meta
    except Exception as exc:
        logger.warning(f"Antigravity CLI generation failed ({exc}). Using offline generator fallback.")

    return generate_offline_metadata(frame_path, song_count)
