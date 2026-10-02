"""Automate downloading the generated video at 720p from Google Flow.

Because Playwright's download interception in persistent contexts is broken 
(the context crashes and throws "Target page closed"), we rely on forcing 
Playwright to download the file into OUTPUT_DIR by setting ``downloads_path``.
Since Playwright crashes before renaming the temporary `.crdownload` file to `.mp4`,
we manually poll for the `.crdownload` file, wait for it to finish growing, and rename it.
"""

import shutil
import time
from pathlib import Path

from loguru import logger
from playwright.sync_api import Page

from automation.config import DOWNLOAD_TIMEOUT, UI_SELECTORS, OUTPUT_DIR
from automation.flow_utils import dismiss_popups


def _find_and_wait_for_crdownload(timeout: int = 60) -> Path | None:
    """Find a recent .crdownload file in OUTPUT_DIR and wait for it to finish.

    Args:
        timeout: Max seconds to wait for the file to appear and finish downloading.

    Returns:
        Path to the completed .crdownload file, or None if not found/timed out.
    """
    start_time = time.time()
    logger.info("Polling for .crdownload file in {}...", OUTPUT_DIR)
    
    # Wait for the file to appear
    crdownload_file = None
    while time.time() - start_time < timeout:
        recent_files = sorted(
            OUTPUT_DIR.glob("*.crdownload"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if recent_files:
            # Pick the most recently modified one
            f = recent_files[0]
            if time.time() - f.stat().st_mtime < 60:
                crdownload_file = f
                break
        time.sleep(1)
        
    if not crdownload_file:
        return None
        
    logger.info("Found active download: {}. Waiting for it to finish...", crdownload_file.name)
    
    # Wait for the file size to stabilize (meaning download is complete)
    last_size = -1
    stable_count = 0
    while time.time() - start_time < timeout:
        try:
            current_size = crdownload_file.stat().st_size
            if current_size == last_size and current_size > 0:
                stable_count += 1
                if stable_count >= 3: # Size unchanged for 3 consecutive polls (3 seconds)
                    logger.info("Download stabilized at {} bytes.", current_size)
                    return crdownload_file
            else:
                last_size = current_size
                stable_count = 0
        except FileNotFoundError:
            # File disappeared? Maybe Playwright managed to rename it to .mp4 natively!
            logger.warning("crdownload file disappeared! It might have been renamed to .mp4 automatically.")
            return None
            
        time.sleep(1)
        
    logger.warning("crdownload file didn't stabilize within timeout.")
    return crdownload_file


def download_video(page: Page, output_path: Path) -> Path:
    """Download the most recently generated video at 720p.

    Args:
        page: The active Playwright page.
        output_path: Destination file path (e.g. ``input/<name>.mp4``).

    Returns:
        The resolved output path where the video was saved.

    Raises:
        PlaywrightTimeout: If the download does not start in time.
        FileNotFoundError: If the file was not successfully downloaded.
    """
    logger.info("Downloading video to: {}", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dismiss_popups(page)

    video_tile = page.locator(UI_SELECTORS["generated_video"]).first
    try:
        video_tile.wait_for(state="visible", timeout=10000)
        logger.info("Opening detail view for the generated video...")
        video_tile.click(force=True)
        page.wait_for_timeout(3000)  # Wait for detail view to animate in
    except Exception as e:
        logger.warning(f"Video tile not visible: {e}")

    dismiss_popups(page)

    # Click the "Download media" button in the top right
    download_btn = page.locator(UI_SELECTORS["download_btn"])
    try:
        download_btn.wait_for(state="visible", timeout=5000)
        download_btn.click()
        page.wait_for_timeout(1000)
    except Exception as e:
        logger.error("Could not find download button! Saving debug DOM and screenshot...")
        page.screenshot(path="error_download_btn.png")
        with open("error_download_dom.html", "w", encoding="utf-8") as f:
            f.write(page.content())
        raise e

    # Click 720p to trigger the Playwright-managed download
    logger.info("Clicking 720p option to trigger native download...")
    
    # We wrap in try-except because clicking this often causes Playwright to throw 
    # "Target page closed" immediately. We catch it and just poll the filesystem.
    try:
        page.locator(UI_SELECTORS["download_720p_option"]).first.click(force=True)
    except Exception as e:
        logger.warning("Clicking 720p threw an error (expected if context closes): {}", e)
    
    # Find the temp file and wait for it to finish
    crdownload = _find_and_wait_for_crdownload(timeout=DOWNLOAD_TIMEOUT)
    
    if crdownload and crdownload.exists():
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(crdownload), str(output_path))
        logger.info("Video correctly saved and renamed to {}", output_path.name)
    else:
        # Fallback: Check if Playwright managed to save an .mp4 after all
        recent_mp4s = sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if recent_mp4s and (time.time() - recent_mp4s[0].stat().st_mtime < 120):
            logger.info("Found an automatically completed .mp4 file.")
            if recent_mp4s[0].resolve() != output_path.resolve():
                if output_path.exists():
                    output_path.unlink()
                shutil.move(str(recent_mp4s[0]), str(output_path))
        else:
            raise FileNotFoundError(
                f"Native download failed: Could not find completed video in {OUTPUT_DIR}"
            )

    # Return to gallery, if page is still alive
    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(1000)
    except Exception:
        pass
    
    return output_path
