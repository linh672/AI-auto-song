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
from automation.flow_utils import dismiss_popups, submit_generation, wait_for_generation


def _configure_video_settings(page: Page) -> None:
    """Select Omni 1.1 Flash, 720p, 8 giây, 16:9, and x1.

    Args:
        page: The active Playwright page.
    """
    dismiss_popups(page)

    # Click settings trigger to open the settings panel
    page.locator(UI_SELECTORS["settings_trigger"]).click()
    page.wait_for_timeout(1000)

    # Find the active settings popover
    popover = page.locator('.cdk-overlay-pane').last
    popover.wait_for(state="visible", timeout=3000)

    # First, switch to Video mode
    video_btn = popover.locator(UI_SELECTORS["settings_video_tab"]).first
    if video_btn.is_visible(timeout=2000):
        video_btn.click(force=True)
        logger.info("Switched to Video mode")
        page.wait_for_timeout(1000)
    else:
        logger.warning("Could not find Video mode button!")

    # Select Frame (Khung hình) tab in settings
    frame_tab = popover.locator(UI_SELECTORS["settings_frame_tab"]).first
    if frame_tab.is_visible(timeout=2000):
        frame_tab.click(force=True)
        page.wait_for_timeout(500)
        logger.info("Switched to Frames mode")
    else:
        logger.warning("Could not find Frame tab in settings panel! Check selector.")


    # Iterate through other settings
    for sel_key, log_msg in [
        ("resolution_720p", "720p"),
        ("duration_8s", "8 giây"),
        ("count_x1", "x1"),
    ]:
        elem = popover.locator(UI_SELECTORS[sel_key]).first
        if elem.is_visible(timeout=1000):
            elem.click()
            logger.info("Selected {}: {}", sel_key, log_msg)
            page.wait_for_timeout(300)
            
    # Close settings panel
    page.keyboard.press("Escape")
    page.wait_for_timeout(500)


def create_video_from_image(page: Page) -> None:
    """Create a looping animation video from the most recent generated image.

    Args:
        page: The active Playwright page.

    Raises:
        PlaywrightTimeout: If video generation does not complete in time.
    """
    logger.info("Creating video from generated image...")

    # Start fresh to clear previous prompts
    page.reload()
    page.locator(UI_SELECTORS["add_ingredients_btn"]).wait_for(state="visible", timeout=30000)
    dismiss_popups(page)

    # Configure settings FIRST so the prompt box switches to Keyframe mode
    _configure_video_settings(page)

    # Now add the Start (Bắt đầu) frame
    logger.info("Adding Start frame...")
    start_btn = page.locator('button:has-text("Start"), button:has-text("Bắt đầu")').first
    if start_btn.is_visible(timeout=2000):
        start_btn.click()
        page.wait_for_timeout(1000)
        
        panel = page.locator('.cdk-overlay-pane').last
        # Switch to Images tab
        img_tab = panel.locator(UI_SELECTORS["asset_tab_images"]).first
        if img_tab.is_visible():
            img_tab.click()
            page.wait_for_timeout(500)
            
        img = panel.locator('[role="option"]').first
        img.click()
        page.wait_for_timeout(500)
        
        add_btn = panel.locator(UI_SELECTORS["add_to_prompt_btn"])
        if add_btn.is_visible():
            add_btn.click()
        page.wait_for_timeout(1000)
    else:
        logger.warning("Could not find Start frame button!")

    # Now add the End (Kết thúc) frame
    logger.info("Adding End frame...")
    end_btn = page.locator('button:has-text("End"), button:has-text("Kết thúc")').first
    if end_btn.is_visible(timeout=2000):
        end_btn.click()
        page.wait_for_timeout(1000)
        
        panel = page.locator('.cdk-overlay-pane').last
        # Switch to Images tab
        img_tab = panel.locator(UI_SELECTORS["asset_tab_images"]).first
        if img_tab.is_visible():
            img_tab.click()
            page.wait_for_timeout(500)
            
        img = panel.locator('[role="option"]').first
        # In the second pass, sometimes clicking unselects it if state is shared, 
        # but in Keyframe mode they are separate slots. We check if it's already selected.
        if img.get_attribute("aria-selected") != "true":
            img.click()
        page.wait_for_timeout(500)
        
        add_btn = panel.locator(UI_SELECTORS["add_to_prompt_btn"])
        if add_btn.is_visible():
            add_btn.click()
        page.wait_for_timeout(1000)
    else:
        logger.warning("Could not find End frame button!")

    # Enter the video prompt
    prompt_area = page.locator(UI_SELECTORS["prompt_paragraph"]).first
    prompt_area.click()
    page.keyboard.type(VIDEO_PROMPT, delay=20)
    logger.info("Entered video prompt.")

    dismiss_popups(page)
    
    submit_generation(page)
    logger.info("Submitted video generation request.")

    # Wait for video to be generated
    wait_for_generation(page, timeout_ms=VIDEO_GENERATION_TIMEOUT * 1000)
    logger.info("Video generated successfully.")
