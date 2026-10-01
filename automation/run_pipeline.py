"""Pipeline orchestrator: process all background images through Google Flow.

Launches a persistent Chrome browser context with anti-bot arguments,
iterates through unprocessed backgrounds, and runs the full
image → video → download workflow for each. Failures are isolated
per-background so one error does not crash the entire pipeline.
"""

import sys
import time
from pathlib import Path

from loguru import logger
from playwright.sync_api import sync_playwright

from automation.config import (
    BACKGROUND_DIR,
    CHARACTER_DIR,
    COOLDOWN_BETWEEN_ITERATIONS,
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
USER_DATA_DIR = r"C:\FlowBotProfile"
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

    with sync_playwright() as pw:
        context = pw.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR,
            channel="chrome",
            headless=False,
            args=CHROME_ARGS,
            accept_downloads=True,
            viewport={"width": 1280, "height": 900},
        )
        page = context.pages[0] if context.pages else context.new_page()

        for i, bg in enumerate(pending, 1):
            logger.info(
                "=== [{}/{}] Processing: {} ===",
                i, len(pending), bg.name,
            )

            try:
                video_name = bg.stem + ".mp4"
                output_path = OUTPUT_DIR / video_name

                create_composite_image(page, character, bg)
                create_video_from_image(page)
                download_video(page, output_path)

                _log_completion(bg.name)
                logger.info("✓ Completed: {}", bg.name)

            except Exception as exc:
                logger.error("✗ Failed {}: {}", bg.name, exc)
                # Save a debug screenshot
                try:
                    screenshot_path = (
                        OUTPUT_DIR / f"error_{bg.stem}.png"
                    )
                    page.screenshot(path=str(screenshot_path))
                    logger.info("Error screenshot saved: {}", screenshot_path)
                except Exception:
                    pass
                errors.append(bg.name)

            # Cooldown between iterations (skip after last one)
            if i < len(pending):
                logger.info("Cooldown: {}s...", COOLDOWN_BETWEEN_ITERATIONS)
                time.sleep(COOLDOWN_BETWEEN_ITERATIONS)

        context.close()

    # Summary
    logger.info("=== Pipeline Complete ===")
    logger.info("Succeeded: {}", len(pending) - len(errors))
    if errors:
        logger.error("Failed ({}): {}", len(errors), ", ".join(errors))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline())
