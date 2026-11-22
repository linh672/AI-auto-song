"""Automate the 'create composite image' step in Google Flow.

Workflow: open the asset panel → add the character image to the prompt →
reopen the asset panel → add the background image to the prompt →
type the image prompt → configure Nano Banana Pro / 16:9 / x1 →
submit → wait for result.
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
from automation.flow_utils import dismiss_popups, submit_generation, wait_for_generation


def _wait_for_asset_upload_ready(page: Page, filename: str, timeout_s: int = 60) -> None:
    """Wait until an uploaded asset finishes uploading and is selectable in the panel.

    Args:
        page: The active Playwright page.
        filename: Uploaded image filename to wait for.
        timeout_s: Maximum seconds to wait for upload completion.
    """
    for _ in range(timeout_s):
        panel = page.locator(".cdk-overlay-pane").last
        asset_option = panel.locator(f'[role="option"]:has-text("{filename}")').first
        if asset_option.count() > 0:
            opt_text = asset_option.inner_text()
            if "Uploading" not in opt_text and "Đang tải lên" not in opt_text:
                return
        page.wait_for_timeout(1000)
    logger.warning("Timed out waiting for '{}' upload spinner to finish.", filename)


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
        _wait_for_asset_upload_ready(page, filename)
        return

    logger.info("Uploading '{}' to Google Flow...", filename)
    upload_btn = page.locator(UI_SELECTORS["upload_media_btn"])
    # The upload button triggers a file chooser
    with page.expect_file_chooser() as fc_info:
        upload_btn.click()
    file_chooser = fc_info.value
    file_chooser.set_files(str(image_path))
    _wait_for_asset_upload_ready(page, filename)
    logger.info("Upload complete: {}", filename)


def _add_single_asset(page: Page, asset_name: str) -> None:
    """Select one asset from the ingredient panel and attach it to the prompt box.

    Ensures the uploaded asset is finished uploading, selects it if not already
    marked with ``asset-item-active``, clicks ``Add to prompt`` when enabled,
    and verifies the ingredient chip is attached to the prompt box.

    Args:
        page: The active Playwright page.
        asset_name: Display name (filename) of the asset to add.
    """
    ingredient_panel = page.locator(".cdk-overlay-pane").last
    if not ingredient_panel.is_visible():
        page.locator(UI_SELECTORS["add_ingredients_btn"]).click()
        page.wait_for_timeout(1000)
        ingredient_panel = page.locator(".cdk-overlay-pane").last
        ingredient_panel.wait_for(state="visible", timeout=3000)

        uploads_tab = ingredient_panel.locator(UI_SELECTORS["asset_tab_uploads"]).first
        if uploads_tab.is_visible(timeout=2000):
            try:
                uploads_tab.click()
                page.wait_for_timeout(1000)
            except Exception as e:
                logger.warning(f"Failed to click Uploads tab: {e}")

    _wait_for_asset_upload_ready(page, asset_name)

    asset_option = ingredient_panel.locator(
        f'[role="option"]:has-text("{asset_name}")'
    ).first
    add_btn = ingredient_panel.locator(UI_SELECTORS["add_to_prompt_btn"]).first

    try:
        asset_option.scroll_into_view_if_needed()
        opt_class = asset_option.get_attribute("class") or ""
        if "asset-item-active" not in opt_class:
            asset_option.click()
            page.wait_for_timeout(800)
    except Exception as e:
        logger.warning(f"Failed to select asset {asset_name}: {e}")

    # Wait for "Add to prompt" button to become enabled, then click it
    if add_btn.is_visible(timeout=3000):
        for _ in range(20):
            if add_btn.get_attribute("disabled") is None:
                break
            page.wait_for_timeout(500)
        try:
            add_btn.click()
            page.wait_for_timeout(1000)
        except Exception as e:
            logger.warning(f"Failed to click Add to Prompt: {e}")

    # Ensure panel is closed if still open
    if ingredient_panel.is_visible():
        try:
            page.keyboard.press("Escape")
            page.wait_for_timeout(500)
        except Exception:
            pass

    # Verify the image ingredient chip is attached in the prompt box
    chip = page.locator('flow-image-ingredient-chip img[alt="Ingredient image"], flow-ingredient-chip')
    if chip.count() > 0:
        logger.info("Verified ingredient chip attached for: {}", asset_name)
    else:
        raise RuntimeError(f"Ingredient chip was not attached to prompt for {asset_name}")

    logger.info("Added asset to prompt: {}", asset_name)


def _open_settings_and_configure(page: Page) -> None:
    """Open the settings panel and select Nano Banana Pro / 16:9 / x1.

    Args:
        page: The active Playwright page.
    """
    dismiss_popups(page)

    # Click settings trigger to open the settings panel (force click to bypass modals)
    try:
        page.locator(UI_SELECTORS["settings_trigger"]).click(force=True)
    except Exception as e:
        logger.warning(f"Failed to click settings trigger: {e}")
    page.wait_for_timeout(1000)

    # Find the active settings popover (Angular Material creates a cdk-overlay-pane at the end of the body)
    popover = page.locator('.cdk-overlay-pane').last
    popover.wait_for(state="visible", timeout=3000)

    # Select Image tab in settings
    image_tab = popover.locator(UI_SELECTORS["settings_image_tab"]).first
    if image_tab.is_visible(timeout=2000):
        image_tab.click(force=True)
        page.wait_for_timeout(800)
    else:
        logger.warning("Could not find Image tab in settings panel! Check selector.")

    # Open model dropdown first
    dropdown_btn = popover.locator(UI_SELECTORS["model_dropdown_btn"]).first
    if dropdown_btn.is_visible(timeout=2000):
        dropdown_btn.click()
        page.wait_for_timeout(500)
        
        # Select Nano Banana Pro from the menu
        model_option = page.locator('.cdk-overlay-pane').last.locator('[role="menuitem"]:has-text("Nano Banana Pro")').first
        if not model_option.is_visible():
            model_option = page.locator('.cdk-overlay-pane').last.locator('text="Nano Banana Pro"').first
            
        if model_option.is_visible():
            model_option.click()
            logger.info("Selected model: Nano Banana Pro")
        else:
            logger.warning("Could not find Nano Banana Pro option in dropdown!")
            page.keyboard.press("Escape")
    else:
        logger.warning("Could not find model dropdown button! Check selector.")
    
    page.wait_for_timeout(500)

    # Select 16:9 ratio
    ratio = popover.locator(UI_SELECTORS["ratio_16_9"]).first
    if ratio.is_visible(timeout=2000):
        ratio.click()
        logger.info("Selected ratio: 16:9")
    else:
        logger.warning("Could not find 16:9 ratio!")

    # Select x1 count
    count = popover.locator(UI_SELECTORS["count_x1"]).first
    if count.is_visible(timeout=2000):
        count.click()
        logger.info("Selected count: x1")

    page.wait_for_timeout(500)


def create_composite_image(
    page: Page,
    # character_path: Path,  # Commented out: dog character workflow disabled
    background_path: Path,
) -> None:
    """Edit a background image in Google Flow using IMAGE_PROMPT.

    Args:
        page: The active Playwright page.
        background_path: Path to the background image to edit.

    Raises:
        PlaywrightTimeout: If generation does not complete in time.
    """
    # logger.info(
    #     "Creating composite: {} + {}",
    #     character_path.name,
    #     background_path.name,
    # )
    logger.info("Editing background image: {}", background_path.name)

    page.goto(FLOW_URL)
    
    # Click New Project if we are on the homepage
    try:
        # Evaluate via JS to avoid Playwright's unicode selector crashes on Windows
        page.wait_for_timeout(3000) # Wait for homepage to load
        clicked = page.evaluate('''() => {
            const btns = Array.from(document.querySelectorAll('button, a, div[role="button"]'));
            const btn = btns.find(b => {
                const txt = b.textContent || '';
                return txt.includes('New project') || txt.includes('Dự án mới');
            });
            if (btn) {
                btn.click();
                return true;
            }
            return false;
        }''')
        if clicked:
            logger.info("Clicked 'New project'")
        else:
            logger.warning("Could not find 'New project' button via JS, saving debug_home.html")
            with open("debug_home.html", "w", encoding="utf-8") as f:
                f.write(page.content())

    except Exception as e:
        logger.warning(f"Failed to click 'New project': {e}")

    # Wait for the general prompt area to load, since add_ingredients_btn might not exist if in Video mode
    page.locator(UI_SELECTORS["prompt_paragraph"]).first.wait_for(state="visible", timeout=30000)
    dismiss_popups(page)

    # First, configure settings to ensure we are in Image mode!
    _open_settings_and_configure(page)
    
    # Close settings
    page.keyboard.press("Escape")
    page.wait_for_timeout(500)
    
    # Now the add ingredients button should be visible
    page.locator(UI_SELECTORS["add_ingredients_btn"]).wait_for(
        state="visible", timeout=10000
    )

    # Upload images if needed (open panel once just for upload check)
    page.locator(UI_SELECTORS["add_ingredients_btn"]).click()
    page.wait_for_timeout(1000)
    
    ingredient_panel = page.locator('.cdk-overlay-pane').last
    ingredient_panel.wait_for(state="visible", timeout=3000)

    uploads_tab = ingredient_panel.locator(UI_SELECTORS["asset_tab_uploads"]).first
    if uploads_tab.is_visible(timeout=2000):
        uploads_tab.click()
        page.wait_for_timeout(1000)

    # _upload_if_needed(page, character_path)
    _upload_if_needed(page, background_path)

    # Attach background image directly from the active panel (dog character image commented out)
    # _add_single_asset(page, character_path.name)
    _add_single_asset(page, background_path.name)

    # Enter prompt (using Shift+Enter for multiline prompts to avoid early Enter submit)
    prompt = page.locator(UI_SELECTORS["prompt_paragraph"]).first
    prompt.click()
    for idx, line in enumerate(IMAGE_PROMPT.splitlines()):
        if idx > 0:
            page.keyboard.press("Shift+Enter")
        page.keyboard.type(line, delay=20)
    logger.info("Entered image prompt.")

    # Submit
    dismiss_popups(page)
    submit_generation(page)
    logger.info("Submitted image generation request.")

    # Wait for result
    wait_for_generation(page, timeout_ms=IMAGE_GENERATION_TIMEOUT * 1000)
    logger.info("Composite image generated successfully.")
