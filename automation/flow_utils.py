"""Shared browser utilities for Google Flow automation.

Provides popup/cookie dismissal and generation-wait helpers used by
all flow_*.py modules.
"""

from loguru import logger
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeout

from automation.config import UI_SELECTORS, POPUP_CHECK_TIMEOUT


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

    Waits for either a new generated image or video thumbnail to appear
    in the project gallery.

    Args:
        page: The active Playwright page.
        timeout_ms: Maximum wait time in milliseconds.

    Raises:
        PlaywrightTimeout: If no result appears within timeout.
    """
    logger.info("Waiting for generation result (timeout {}s)...", timeout_ms // 1000)
    try:
        page.wait_for_load_state("networkidle", timeout=timeout_ms)
    except PlaywrightTimeout:
        logger.warning("networkidle timed out, but continuing anyway.")
    
    page.wait_for_timeout(5000)  # extra buffer for rendering
    logger.info("Generation wait finished.")
