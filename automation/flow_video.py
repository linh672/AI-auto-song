"""Automate the 'create video from image' step in Google Flow.

Takes the most recently generated image, adds it to the prompt,
configures Omni 1.1 Flash / 720p / 8s / 16:9, and generates a video.
"""

from loguru import logger
from playwright.sync_api import Page

from automation.config import (
    FLOW_URL,
    UI_SELECTORS,
    VIDEO_GENERATION_TIMEOUT,
    VIDEO_PROMPT,
)
from automation.flow_utils import dismiss_popups, wait_for_generation


def _configure_video_settings(page: Page) -> None:
    """Select Omni 1.1 Flash, 720p, 8 giây, 16:9, and x1.

    Args:
        page: The active Playwright page.
    """
    dismiss_popups(page)

    # Click settings trigger to open the settings panel
    page.locator(UI_SELECTORS["settings_trigger"]).click()
    page.wait_for_timeout(1000)

    # Select Video tab in settings
    video_tab = page.locator(UI_SELECTORS["settings_video_tab"]).first
    if video_tab.is_visible(timeout=2000):
        video_tab.click()
        page.wait_for_timeout(500)

    # Try to select the model
    # The button text shows the current model. If we click it, it opens the dropdown.
    # We should only click the dropdown option if we need to change it, but it's tricky to 
    # know without clicking the button first. Let's press Escape to ensure any open dropdowns are closed.
    model_btn = page.locator(UI_SELECTORS["omni_flash"]).first
    if model_btn.is_visible(timeout=2000):
        model_btn.click()
        page.wait_for_timeout(500)
        # Click it again if it's an option in the dropdown
        dropdown_option = page.locator('[role="option"]:has-text("Omni 1.1 Flash")')
        if dropdown_option.is_visible():
            dropdown_option.click()
        page.keyboard.press("Escape")  # Close dropdown if it's still open
        logger.info("Selected model: Omni 1.1 Flash")
        page.wait_for_timeout(500)

    # Iterate through other settings (video has no aspect ratio)
    for sel_key, log_msg in [
        ("resolution_720p", "720p"),
        ("duration_8s", "8 giây"),
        ("count_x1", "x1"),
    ]:
        elem = page.locator(UI_SELECTORS[sel_key]).first
        if elem.is_visible(timeout=1000):
            elem.click()
            logger.info("Selected {}: {}", sel_key, log_msg)
            page.wait_for_timeout(300)


def create_video_from_image(page: Page) -> None:
    """Create a looping animation video from the most recent generated image.

    Args:
        page: The active Playwright page.

    Raises:
        PlaywrightTimeout: If video generation does not complete in time.
    """
    logger.info("Creating video from generated image...")

    # Start fresh to clear previous prompts
    page.goto(FLOW_URL)
    page.locator(UI_SELECTORS["add_ingredients_btn"]).wait_for(state="visible", timeout=30000)
    dismiss_popups(page)

    # Open the ingredients panel
    page.locator(UI_SELECTORS["add_ingredients_btn"]).click()
    page.wait_for_timeout(1000)

    # Switch to Images tab to ensure we pick an image, not a video
    images_tab = page.locator(UI_SELECTORS["asset_tab_images"]).first
    if images_tab.is_visible(timeout=2000):
        images_tab.click()
        page.wait_for_timeout(1000)

    # Pick the most recent image (first in the list)
    first_image = page.locator(UI_SELECTORS["asset_list"]).locator('[role="option"]').first
    first_image.click()
    page.wait_for_timeout(500)

    # Add to prompt (sometimes the panel auto-closes, so make it optional)
    add_btn = page.locator(UI_SELECTORS["add_to_prompt_btn"])
    if add_btn.is_visible(timeout=2000):
        add_btn.click()
        page.wait_for_timeout(1000)
    logger.info("Added recent generated image to prompt.")

    # Enter the video prompt
    prompt_area = page.locator(UI_SELECTORS["prompt_paragraph"]).first
    prompt_area.click()
    page.keyboard.type(VIDEO_PROMPT, delay=20)
    logger.info("Entered video prompt.")

    _configure_video_settings(page)

    dismiss_popups(page)
    
    page.locator(UI_SELECTORS["start_generation_btn"]).click()
    logger.info("Submitted video generation request.")

    # Wait for video to be generated
    wait_for_generation(page, timeout_ms=VIDEO_GENERATION_TIMEOUT * 1000)
    logger.info("Video generated successfully.")
