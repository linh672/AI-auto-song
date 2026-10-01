"""Automate the 'create composite image' step in Google Flow.

Workflow: open the asset panel → add the character image, then the
background image to the prompt → type the image prompt → configure
Nano Banana Pro / 16:9 / x1 settings → submit → wait for result.
"""

from pathlib import Path

from loguru import logger
from playwright.sync_api import Page

from automation.config import (
    FLOW_URL,
    IMAGE_GENERATION_TIMEOUT,
    IMAGE_PROMPT,
    UI_SELECTORS,
)
from automation.flow_utils import dismiss_popups, wait_for_generation


def _upload_if_needed(page: Page, image_path: Path) -> None:
    """Upload an image to Google Flow if it is not already in the asset list.

    Args:
        page: The active Playwright page.
        image_path: Local path to the image file.
    """
    filename = image_path.name
    asset_option = page.locator(
        f'[role="option"]:has-text("{filename}")'
    )
    if asset_option.count() > 0:
        logger.info("Asset '{}' already uploaded.", filename)
        return

    logger.info("Uploading '{}' to Google Flow...", filename)
    upload_btn = page.locator(UI_SELECTORS["upload_media_btn"])
    # The upload button triggers a file chooser
    with page.expect_file_chooser() as fc_info:
        upload_btn.click()
    file_chooser = fc_info.value
    file_chooser.set_files(str(image_path))
    page.wait_for_timeout(3000)  # wait for upload to complete
    logger.info("Upload complete: {}", filename)


def _add_assets_to_prompt(page: Page, asset_names: list[str]) -> None:
    """Select multiple assets from the ingredient panel and add them to the prompt.

    Args:
        page: The active Playwright page.
        asset_names: List of display names (filenames) to add.
    """
    for asset_name in asset_names:
        asset_option = page.locator(f'[role="option"]:has-text("{asset_name}")').first
        asset_option.click()
        page.wait_for_timeout(500)

    # Click "Add to prompt" once for all selected assets
    page.locator(UI_SELECTORS["add_to_prompt_btn"]).click()
    page.wait_for_timeout(1000)
    logger.info("Added assets to prompt: {}", ", ".join(asset_names))


def _open_settings_and_configure(page: Page) -> None:
    """Open the settings panel and select Nano Banana Pro / 16:9 / x1.

    Args:
        page: The active Playwright page.
    """
    dismiss_popups(page)

    # Click settings trigger to open the settings panel
    page.locator(UI_SELECTORS["settings_trigger"]).click()
    page.wait_for_timeout(1000)

    # Select Image tab in settings
    image_tab = page.locator(UI_SELECTORS["settings_image_tab"]).first
    if image_tab.is_visible(timeout=2000):
        image_tab.click()
        page.wait_for_timeout(500)

    # Select Nano Banana Pro model
    model = page.locator(UI_SELECTORS["nano_banana_pro"])
    if model.is_visible(timeout=2000):
        model.click()
        logger.info("Selected model: Nano Banana Pro")
        page.wait_for_timeout(500)

    # Select 16:9 ratio
    ratio = page.locator(UI_SELECTORS["ratio_16_9"]).first
    if ratio.is_visible(timeout=2000):
        ratio.click()
        logger.info("Selected ratio: 16:9")

    # Select x1 count
    count = page.locator(UI_SELECTORS["count_x1"]).first
    if count.is_visible(timeout=2000):
        count.click()
        logger.info("Selected count: x1")

    page.wait_for_timeout(500)


def create_composite_image(
    page: Page,
    character_path: Path,
    background_path: Path,
) -> None:
    """Create a composite image by combining character + background in Google Flow.

    Args:
        page: The active Playwright page.
        character_path: Path to the character (dog) image.
        background_path: Path to the background image.

    Raises:
        PlaywrightTimeout: If generation does not complete in time.
    """
    logger.info(
        "Creating composite: {} + {}",
        character_path.name,
        background_path.name,
    )

    page.goto(FLOW_URL)
    page.locator(UI_SELECTORS["add_ingredients_btn"]).wait_for(state="visible", timeout=30000)
    dismiss_popups(page)

    # Open the ingredients panel
    page.locator(UI_SELECTORS["add_ingredients_btn"]).click()
    page.wait_for_timeout(1000)

    # Switch to Uploads tab so we can find/upload our images
    uploads_tab = page.locator(UI_SELECTORS["asset_tab_uploads"]).first
    if uploads_tab.is_visible(timeout=2000):
        uploads_tab.click()
        page.wait_for_timeout(1000)

    # Upload images if not already in Flow
    _upload_if_needed(page, character_path)
    _upload_if_needed(page, background_path)

    # Add both character image and background image at once
    _add_assets_to_prompt(page, [character_path.name, background_path.name])

    # Type the prompt
    prompt = page.locator(UI_SELECTORS["prompt_paragraph"]).first
    prompt.click()
    page.keyboard.type(IMAGE_PROMPT, delay=20)
    logger.info("Entered image prompt.")

    # Configure settings
    _open_settings_and_configure(page)

    # Submit
    dismiss_popups(page)
    page.locator(UI_SELECTORS["start_generation_btn"]).click()
    logger.info("Submitted image generation request.")

    # Wait for result
    wait_for_generation(page, timeout_ms=IMAGE_GENERATION_TIMEOUT * 1000)
    logger.info("Composite image generated successfully.")
