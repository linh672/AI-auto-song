"""Pipeline orchestrator: process all background images through Google Flow.

Launches a persistent Chrome browser context with anti-bot arguments,
iterates through unprocessed backgrounds, and runs the full
image → video → download workflow for each. Failures are isolated
per-background so one error does not crash the entire pipeline.
"""

import subprocess
import sys
import time
from pathlib import Path

from loguru import logger
from playwright.sync_api import sync_playwright

from automation.config import (
    BACKGROUND_DIR,
    CHARACTER_DIR,
    COOLDOWN_BETWEEN_ITERATIONS,
    FLOW_URL,
    IMAGE_EXTENSIONS,
    OUTPUT_DIR,
    PROGRESS_LOG,
)
from automation.flow_download import download_video
from automation.flow_image import create_composite_image
from automation.flow_video import create_video_from_image

# ---------------------------------------------------------------------------
# Browser configuration (Windows-specific)
# ---------------------------------------------------------------------------
USER_DATA_DIR = r"C:\FlowBotProfile_Account3"
CHROME_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--disable-popup-blocking",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kill_chrome() -> None:
    """Force-kill all Chrome processes to release the profile lock."""
    subprocess.run(
        ["taskkill", "/F", "/IM", "chrome.exe", "/T"],
        capture_output=True,
    )


def _safe_close(ctx) -> None:
    """Close a browser context, ignoring errors if already closed."""
    try:
        ctx.close()
    except Exception:
        pass

def _get_character_image() -> Path:
    """Find the single character image in the character directory.

    Returns:
        Path to the character image.

    Raises:
        FileNotFoundError: If no image is found in the character directory.
    """
    for f in CHARACTER_DIR.iterdir():
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            logger.info("Character image: {}", f.name)
            return f
    raise FileNotFoundError(f"No image found in {CHARACTER_DIR}")


def _get_background_images() -> list[Path]:
    """List all background images sorted by name.

    Returns:
        Sorted list of background image paths.
    """
    images = [
        f for f in BACKGROUND_DIR.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def _get_completed() -> set[str]:
    """Read progress.log to find already-processed backgrounds.

    Returns:
        Set of background filenames that have been completed.
    """
    if not PROGRESS_LOG.exists():
        return set()
    lines = PROGRESS_LOG.read_text(encoding="utf-8").strip().splitlines()
    return {line.strip() for line in lines if line.strip()}


def _log_completion(background_name: str) -> None:
    """Append a completed background filename to progress.log.

    Args:
        background_name: Filename of the processed background.
    """
    with PROGRESS_LOG.open("a", encoding="utf-8") as f:
        f.write(background_name + "\n")
    logger.info("Logged completion: {}", background_name)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> int:
    """Orchestrate the full automation pipeline.

    Launches a persistent Chrome browser context that reuses cookies and
    login state from ``C:\\FlowBotProfile``. For each unprocessed background,
    runs create_image → create_video → download, isolated in a try-except
    so a single failure does not crash the run.

    Returns:
        Exit code: 0 on full success, 1 if any background failed.
    """
    character = _get_character_image()
    backgrounds = _get_background_images()
    completed = _get_completed()

    pending = [bg for bg in backgrounds if bg.name not in completed]

    logger.info("Total backgrounds: {}", len(backgrounds))
    logger.info("Already completed: {}", len(completed))
    logger.info("Pending: {}", len(pending))

    if not pending:
        logger.info("All backgrounds already processed. Nothing to do.")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for i, bg in enumerate(pending, 1):
        logger.info(
            "=== [{}/{}] Processing: {} ===",
            i, len(pending), bg.name,
        )
        try:
            output_path = OUTPUT_DIR / (bg.stem + ".mp4")
            _kill_chrome()

            # Phase 1: Generate image + video
            with sync_playwright() as pw:
                ctx = pw.chromium.launch_persistent_context(
                    user_data_dir=USER_DATA_DIR, channel="chrome",
                    headless=False, args=CHROME_ARGS,
                    viewport={"width": 1280, "height": 900},
                    accept_downloads=True, downloads_path=str(OUTPUT_DIR)
                )
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                create_composite_image(page, character, bg)
                create_video_from_image(page)
                project_url = page.url
                _safe_close(ctx)

            _kill_chrome()
            time.sleep(2)

            # Phase 2: Download with a fresh browser context
            with sync_playwright() as pw:
                ctx = pw.chromium.launch_persistent_context(
                    user_data_dir=USER_DATA_DIR, channel="chrome",
                    headless=False, args=CHROME_ARGS,
                    accept_downloads=True,
                    downloads_path=str(OUTPUT_DIR.resolve()),
                    viewport={"width": 1280, "height": 900},
                )
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                page.goto(project_url)
                page.wait_for_timeout(5000)
                download_video(page, output_path)
                _safe_close(ctx)

            _kill_chrome()
            _log_completion(bg.name)
            logger.info("✓ Completed: {}", bg.name)

        except Exception as exc:
            logger.error("✗ Failed {}: {}", bg.name, exc)
            _kill_chrome()
            errors.append(bg.name)

        # Cooldown between iterations (skip after last one)
        if i < len(pending):
            logger.info("Cooldown: {}s...", COOLDOWN_BETWEEN_ITERATIONS)
            time.sleep(COOLDOWN_BETWEEN_ITERATIONS)

    # Summary
    logger.info("=== Pipeline Complete ===")
    logger.info("Succeeded: {}", len(pending) - len(errors))
    if errors:
        logger.error("Failed ({}): {}", len(errors), ", ".join(errors))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline())
