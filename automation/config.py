"""Centralized configuration constants and UI selectors for Google Flow automation.

All Playwright locators live in UI_SELECTORS. Operational scripts
(flow_image, flow_video, flow_download) import this dictionary and
never hardcode UI text.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_AUTOMATION_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _AUTOMATION_DIR.parent

CHARACTER_DIR: Path = _AUTOMATION_DIR / "character"
BACKGROUND_DIR: Path = _AUTOMATION_DIR / "background"
GENERATED_IMAGES_DIR: Path = _AUTOMATION_DIR / "generated_images"
OUTPUT_DIR: Path = _REPO_ROOT / "input"
PROGRESS_LOG: Path = _AUTOMATION_DIR / "progress.log"

# ---------------------------------------------------------------------------
# Google Flow URL
# ---------------------------------------------------------------------------
FLOW_URL = "https://flow.google.com"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
IMAGE_PROMPT = (
    "add the first dog in the first picture into the second picture "
    "(the new picture ratio 16:9) and change the pose so the dog is "
    "more chill dont add any text"
)

VIDEO_PROMPT = (
    "A seamless, subtle looping animation. The cute black puppy rests "
    "quietly on the ground in the meadow, showing only gentle, natural "
    "chest breathing and an occasional slow, sleepy blink, keeping its "
    "mouth completely still."
)

# ---------------------------------------------------------------------------
# Timeouts (seconds unless noted)
# ---------------------------------------------------------------------------
IMAGE_GENERATION_TIMEOUT = 120   # 2 minutes
VIDEO_GENERATION_TIMEOUT = 300   # 5 minutes
DOWNLOAD_TIMEOUT = 180            # 3 minutes
POPUP_CHECK_TIMEOUT = 500        # milliseconds
COOLDOWN_BETWEEN_ITERATIONS = 15  # seconds

# ---------------------------------------------------------------------------
# Retry settings
# ---------------------------------------------------------------------------
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# ---------------------------------------------------------------------------
# UI Selectors — calibrated from actual Google Flow DOM inspection.
#
# Operational scripts import UI_SELECTORS and never hardcode UI text.
# If the Google Flow UI changes, update only this dictionary.
# ---------------------------------------------------------------------------
UI_SELECTORS = {
    # --- Cookie consent (fresh sessions) ---
    "cookie_ok": 'button:has-text("OK, got it")',

    # --- Homepage ---
    "new_project_btn": 'text=/(New project|Dự án mới)/i',

    # --- Prompt area (bottom bar) ---
    "prompt_paragraph": '[contenteditable="true"]',
    "prompt_placeholder": 'text="What do you want to create?"',

    # --- Add images to prompt ---
    "add_ingredients_btn": 'button[aria-label="Add ingredients to the prompt box"]',
    "upload_media_btn": 'button:has-text("Upload media"), button:has-text("Tải phương tiện lên")',
    "asset_list": '[role="listbox"][aria-label="Asset list"]',
    "add_to_prompt_btn": 'button:has-text("Add to prompt"), button:has-text("Thêm vào lời nhắc")',
    "close_ingredients_btn": 'button[aria-label="Add ingredients to the prompt box"][expanded]',

    # --- Asset panel tabs ---
    "asset_tab_all": '[role="tab"]:has(:text-is("All"))',
    "asset_tab_images": '[role="tab"]:has(:text-is("Images")), [role="tab"]:has(:text-is("Hình ảnh"))',
    "asset_tab_uploads": '[role="tab"]:has(:text-is("Uploads")), [role="tab"]:has(:text-is("Nội dung tải lên"))',
    "asset_search": 'textbox[aria-label="Search assets"]',

    # --- Generation ---
    "start_generation_btn": 'button[aria-label="Start generation"]',

    # --- Settings trigger (opens settings panel) ---
    "settings_trigger": 'button[aria-label="Settings trigger"]',

    # --- Settings panel options (text-based selectors) ---
    "model_dropdown_btn": 'button[aria-label="Select model family"]',
    "nano_banana_pro": 'text="Nano Banana Pro"',
    "nano_banana_2": 'text="Nano Banana 2", text="Imagen 3"',
    "omni_flash": 'text="Omni 1.1 Flash"',
    "ratio_16_9": 'text="16:9"',
    "resolution_720p": 'text="720p"',
    "duration_8s": 'text="8 giây", text="8s"',
    "count_x1": 'text="x1"',

    # --- Settings panel tabs ---
    # Angular Material buttons contain <mat-icon>ligature_name</mat-icon> which pollutes the button's innerText.
    # We must target the exact text span inside the button.
    "settings_image_tab": '.toggle-text:text-is("Image"), .toggle-text:text-is("Hình ảnh")',
    "settings_video_tab": '.toggle-text:text-is("Video")',
    "settings_frame_tab": '.toggle-text:text-is("Frames"), .toggle-text:text-is("Khung hình")',
    "settings_component_tab": '.toggle-text:text-is("Component"), .toggle-text:text-is("Thành phần")',

    # --- Generated results ---
    "generated_image": 'img[alt="Tile displaying a user\'s image"]',
    "generated_video": 'img[alt="Generated video thumbnail"]',

    # --- Media detail view (after clicking a tile) ---
    "download_btn": 'button[aria-label="Download media"]',
    "download_720p_option": '[role="menuitem"]:has-text("720p")',

    # --- Popups / overlays ---
    "dismiss_overlay": 'button[aria-label="Close"], button[aria-label="Đóng"]',
    "dismiss_x_btn": 'button:has-text("✕")',
    "error_modal": 'text="Không thành công", text="Failed", text="Error"',
    "error_dismiss_btn": 'button:has-text("OK")',

    # --- Navigation ---
    "home_btn": 'button[aria-label="Home"]',
    "nav_all_media": 'text="All media"',
}
