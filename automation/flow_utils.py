"""Shared browser utilities for Google Flow automation.

Provides popup/cookie dismissal and generation-wait helpers used by
all flow_*.py modules.
"""

import time
from loguru import logger
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeout

from automation.config import UI_SELECTORS, POPUP_CHECK_TIMEOUT


class InsufficientCreditsError(RuntimeError):
    """Raised when Google Flow credits are depleted."""


def submit_generation(page: Page) -> None:
    """Click the Start generation button, failing fast if credits are depleted.

    Args:
        page: The active Playwright page.

    Raises:
        InsufficientCreditsError: If the credits warning button is shown.
        PlaywrightTimeout: If the button is not found for another reason.
    """
    warning_btn = page.locator('button.prompt-warning-button')
    if warning_btn.is_visible(timeout=500):
        raise InsufficientCreditsError(
            "Google Flow credits are depleted. Top up at flow.google.com."
        )
    page.locator(UI_SELECTORS["start_generation_btn"]).click()


def dismiss_popups(page: Page) -> None:
    """Dismiss unexpected overlays, cookie banners, and error modals.

    Scans known popup selectors and clicks dismiss buttons when visible.
    Uses a short timeout so it never blocks the happy path.

    Args:
        page: The active Playwright page.
    """
    dismiss_selectors = [
        ("cookie_ok", UI_SELECTORS["cookie_ok"]),
        ("dismiss_overlay", UI_SELECTORS["dismiss_overlay"]),
        ("dismiss_x_btn", UI_SELECTORS["dismiss_x_btn"]),
        ("error_dismiss_btn", UI_SELECTORS["error_dismiss_btn"]),
    ]

    for name, selector in dismiss_selectors:
        try:
            element = page.locator(selector).first
            if element.is_visible(timeout=POPUP_CHECK_TIMEOUT):
                element.click()
                logger.info("Dismissed popup via '{}'", name)
                page.wait_for_timeout(300)
        except PlaywrightTimeout:
            pass
        except Exception as exc:
            logger.debug("Popup check '{}' raised: {}", name, exc)


def wait_for_generation(page: Page, timeout_ms: int) -> None:
    """Wait for new content to appear after submitting a generation.

    Waits for the progress percentage (e.g. 7%, 28%) to appear and then disappear.

    Args:
        page: The active Playwright page.
        timeout_ms: Maximum wait time in milliseconds.

    Raises:
        PlaywrightTimeout: If no result appears within timeout.
    """
    logger.info("Waiting for generation result (timeout {}s)...", timeout_ms // 1000)
    
    # We will poll for the absence of any elements containing "%" inside the gallery.
    start_time = time.time()
    
    # First, wait for the progress indicator to APPEAR (up to 30 seconds)
    logger.info("Waiting for progress indicator to appear...")
    appeared = False
    while time.time() - start_time < 30:
        progress_visible = page.evaluate('''() => {
            const tiles = document.querySelectorAll('*');
            for (let tile of tiles) {
                if (tile.textContent && tile.textContent.trim().match(/^\\d{1,3}%$/)) {
                    return true;
                }
            }
            return false;
        }''')
        if progress_visible:
            appeared = True
            logger.info("Progress indicator found.")
            break
        page.wait_for_timeout(1000)
        
    if not appeared:
        logger.warning("No progress indicator appeared within 30s. Assuming it either finished instantly or failed.")
        
    # Now, wait for it to DISAPPEAR
    logger.info("Waiting for generation to finish...")
    while time.time() - start_time < timeout_ms / 1000:
        progress_visible = page.evaluate('''() => {
            const tiles = document.querySelectorAll('*');
            for (let tile of tiles) {
                if (tile.textContent && tile.textContent.trim().match(/^\\d{1,3}%$/)) {
                    return true;
                }
            }
            return false;
        }''')
        
        if not progress_visible:
            logger.info("No active progress indicators found. Generation complete.")
            page.wait_for_timeout(2000)
            return
            
        page.wait_for_timeout(3000)
        logger.info("Still generating...")
        
    logger.warning("Generation wait timed out! Proceeding anyway.")
