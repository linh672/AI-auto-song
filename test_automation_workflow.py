"""Unit tests for the updated Google Flow girl image-edit and loop-video workflow."""

from pathlib import Path
import unittest
from unittest.mock import MagicMock, call, patch

import automation.config as config
from automation.flow_image import _add_single_asset, _wait_for_asset_upload_ready, create_composite_image
from automation.flow_video import create_video_from_image
from automation.run_pipeline import run_pipeline


class TestAutomationWorkflow(unittest.TestCase):
    """Verify prompt constants, single-image edit step, video step, and pipeline orchestration."""

    def test_prompts_match_girl_workflow(self) -> None:
        """Verify IMAGE_PROMPT and VIDEO_PROMPT match the new girl workflow and CHARACTER_DIR is disabled."""
        expected_image_prompt = (
            "change the color of the hair of the girl\n"
            "change the weather to midnight and make it deeply emotional"
        )
        self.assertEqual(config.IMAGE_PROMPT, expected_image_prompt)
        self.assertEqual(config.VIDEO_PROMPT, "animate the girl as a continuous loop")
        self.assertFalse(hasattr(config, "CHARACTER_DIR"))

    @patch("automation.flow_image.wait_for_generation")
    @patch("automation.flow_image.submit_generation")
    @patch("automation.flow_image.dismiss_popups")
    @patch("automation.flow_image._add_single_asset")
    @patch("automation.flow_image._upload_if_needed")
    @patch("automation.flow_image._open_settings_and_configure")
    def test_create_composite_image_uses_only_background(
        self,
        mock_settings: MagicMock,
        mock_upload: MagicMock,
        mock_add_asset: MagicMock,
        mock_dismiss: MagicMock,
        mock_submit: MagicMock,
        mock_wait: MagicMock,
    ) -> None:
        """Verify Stage 1 uploads/adds only the background image and types multiline IMAGE_PROMPT with Shift+Enter."""
        page = MagicMock()
        page.evaluate.return_value = True
        bg_path = Path("automation/background/girl_scene.jpg")

        create_composite_image(page, bg_path)

        mock_settings.assert_called_once_with(page)
        mock_upload.assert_called_once_with(page, bg_path)
        mock_add_asset.assert_called_once_with(page, "girl_scene.jpg")
        self.assertEqual(
            page.keyboard.type.call_args_list,
            [
                call("change the color of the hair of the girl", delay=20),
                call("change the weather to midnight and make it deeply emotional", delay=20),
            ],
        )
        self.assertIn(call("Shift+Enter"), page.keyboard.press.call_args_list)
        mock_submit.assert_called_once_with(page)
        mock_wait.assert_called_once()

    def test_upload_ready_and_add_single_asset_waits_for_enabled_button(self) -> None:
        """Verify _wait_for_asset_upload_ready polls until Uploading finishes and _add_single_asset attaches chip."""
        page = MagicMock()
        panel = MagicMock()
        panel.is_visible.return_value = True
        page.locator.return_value.last = panel
        page.locator.return_value.count.return_value = 1

        asset_opt = MagicMock()
        asset_opt.count.return_value = 1
        asset_opt.inner_text.side_effect = ["Uploading\ntest.jpg\nImage", "test.jpg\nImage", "test.jpg\nImage"]
        asset_opt.get_attribute.return_value = "asset-item"

        add_btn = MagicMock()
        add_btn.is_visible.return_value = True
        add_btn.get_attribute.side_effect = ["true", None, None]

        def panel_locator(selector: str) -> MagicMock:
            loc = MagicMock()
            if '[role="option"]' in selector:
                loc.first = asset_opt
            else:
                loc.first = add_btn
            return loc

        panel.locator.side_effect = panel_locator

        _wait_for_asset_upload_ready(page, "test.jpg", timeout_s=5)
        _add_single_asset(page, "test.jpg")

        asset_opt.click.assert_called_once()
        add_btn.click.assert_called_once()

    @patch("automation.flow_video.wait_for_generation")
    @patch("automation.flow_video.submit_generation")
    @patch("automation.flow_video.dismiss_popups")
    @patch("automation.flow_video._configure_video_settings")
    def test_create_video_from_image_types_loop_prompt(
        self,
        mock_video_settings: MagicMock,
        mock_dismiss: MagicMock,
        mock_submit: MagicMock,
        mock_wait: MagicMock,
    ) -> None:
        """Verify Stage 2 configures video keyframes and types the continuous loop VIDEO_PROMPT."""
        page = MagicMock()

        create_video_from_image(page)

        mock_video_settings.assert_called_once_with(page)
        page.keyboard.type.assert_called_once_with(
            "animate the girl as a continuous loop",
            delay=20,
        )
        mock_submit.assert_called_once_with(page)
        mock_wait.assert_called_once()

    @patch("automation.run_pipeline.download_video")
    @patch("automation.run_pipeline.create_video_from_image")
    @patch("automation.run_pipeline.create_composite_image")
    @patch("automation.run_pipeline.sync_playwright")
    @patch("automation.run_pipeline._kill_chrome")
    @patch("automation.run_pipeline._log_completion")
    @patch("automation.run_pipeline._get_completed", return_value=set())
    @patch("automation.run_pipeline._get_background_images")
    def test_run_pipeline_executes_without_character_image(
        self,
        mock_get_bgs: MagicMock,
        mock_get_completed: MagicMock,
        mock_log_completion: MagicMock,
        mock_kill_chrome: MagicMock,
        mock_sync_pw: MagicMock,
        mock_create_img: MagicMock,
        mock_create_vid: MagicMock,
        mock_download: MagicMock,
    ) -> None:
        """Verify run_pipeline passes only (page, bg) to create_composite_image without a dog character image."""
        bg_path = Path("automation/background/girl_scene.jpg")
        mock_get_bgs.return_value = [bg_path]

        mock_page = MagicMock()
        mock_page.url = "https://flow.google.com/project/test-123"
        mock_ctx = MagicMock()
        mock_ctx.pages = [mock_page]
        mock_pw_instance = MagicMock()
        mock_pw_instance.chromium.launch_persistent_context.return_value = mock_ctx
        mock_sync_pw.return_value.__enter__.return_value = mock_pw_instance

        with patch("automation.run_pipeline.time.sleep"):
            exit_code = run_pipeline()

        self.assertEqual(exit_code, 0)
        mock_create_img.assert_called_once_with(mock_page, bg_path)
        mock_create_vid.assert_called_once_with(mock_page)
        mock_download.assert_called_once()
        mock_log_completion.assert_called_once_with("girl_scene.jpg")


if __name__ == "__main__":
    unittest.main()
