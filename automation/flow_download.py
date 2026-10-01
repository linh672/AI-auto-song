"""Automate downloading the generated video at 720p from Google Flow.

Uses Playwright's ``page.expect_download()`` context manager to intercept
the browser download event.
"""

from pathlib import Path

from loguru import logger
from playwright.sync_api import Page

from automation.config import FLOW_URL, DOWNLOAD_TIMEOUT, UI_SELECTORS
from automation.flow_utils import dismiss_popups


def download_video(page: Page, output_path: Path) -> Path:
    """Download the most recently generated video at 720p.

    Args:
        page: The active Playwright page.
        output_path: Destination file path (e.g. ``input/<name>.mp4``).

    Returns:
        The resolved output path where the video was saved.

    Raises:
        PlaywrightTimeout: If the download does not start in time.
    """
    logger.info("Downloading video to: {}", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Start fresh in the gallery view
    page.goto(FLOW_URL)
    page.locator(UI_SELECTORS["add_ingredients_btn"]).wait_for(state="visible", timeout=30000)
    dismiss_popups(page)

    # Click the most recent generated video tile to open detail view
    video_tile = page.locator(UI_SELECTORS["generated_video"]).first
    video_tile.click()
    page.wait_for_timeout(2000)  # Wait for detail view to animate in

    dismiss_popups(page)

    # Click the "Download media" button in the top right
    page.locator(UI_SELECTORS["download_btn"]).click()
    page.wait_for_timeout(1000)

    # Intercept the download event and click 720p
    with page.expect_download(timeout=DOWNLOAD_TIMEOUT * 1000) as download_info:
        page.locator(UI_SELECTORS["download_720p_option"]).first.click()

    download = download_info.value
    download.save_as(str(output_path))

    logger.info("Video saved: {}", output_path)
    return output_path
