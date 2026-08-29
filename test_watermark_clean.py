# -*- coding: utf-8 -*-
"""test_watermark_clean.py - Unit tests for the watermark mitigation modules.

Coverage
--------
- _butterworth_bandstop_mask  : shape, value range, DC preservation, mid-freq attenuation.
- apply_fft_notch_to_frame    : output shape, dtype, identity on flat frames.
- strip_container_metadata    : ffmpeg success / failure fallback paths, flag verification.
- deep_clean_frames           : dispatcher routing, error fallback to original path.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np

from watermark_clean import deep_clean_frames, strip_container_metadata
from watermark_fft import _butterworth_bandstop_mask, apply_fft_notch_to_frame


# ---------------------------------------------------------------------------
# FFT mask generation
# ---------------------------------------------------------------------------


class TestButterworthMask(unittest.TestCase):
    """FFT Butterworth band-stop mask generation."""

    def test_mask_shape_landscape(self):
        """Mask shape must match the requested frame dimensions."""
        mask = _butterworth_bandstop_mask(1080, 1920)
        self.assertEqual(mask.shape, (1080, 1920))

    def test_mask_shape_portrait(self):
        mask = _butterworth_bandstop_mask(1920, 1080)
        self.assertEqual(mask.shape, (1920, 1080))

    def test_dc_component_not_attenuated(self):
        """Centre pixel (DC component) must pass through — mask ≈ 1.0 at origin."""
        mask = _butterworth_bandstop_mask(64, 64)
        self.assertAlmostEqual(float(mask[32, 32]), 1.0, places=2)

    def test_mid_frequency_attenuated(self):
        """Mid-frequency region inside the notch band must be attenuated by depth."""
        mask = _butterworth_bandstop_mask(64, 64, low_cut=0.10, high_cut=0.40, depth=0.06)
        # Pixel at radius ~0.25 Nyquist is in the centre of the notch band.
        self.assertLess(float(mask[32, 40]), 1.0)
        self.assertGreaterEqual(float(mask[32, 40]), 1.0 - 0.06 - 0.01)

    def test_values_in_unit_range(self):
        """All mask values must lie in [0.0, 1.0]."""
        mask = _butterworth_bandstop_mask(64, 64)
        self.assertGreaterEqual(float(mask.min()), 0.0)
        self.assertLessEqual(float(mask.max()), 1.0)

    def test_dtype_float64(self):
        mask = _butterworth_bandstop_mask(32, 32)
        self.assertEqual(mask.dtype, np.float64)


# ---------------------------------------------------------------------------
# Per-frame FFT notch filter
# ---------------------------------------------------------------------------


class TestApplyFftNotchToFrame(unittest.TestCase):
    """Per-frame FFT Butterworth band-stop filter."""

    def _random_frame(self, h: int = 64, w: int = 64) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)

    def test_output_shape_preserved(self):
        frame = self._random_frame()
        mask = _butterworth_bandstop_mask(*frame.shape[:2])
        result = apply_fft_notch_to_frame(frame, mask)
        self.assertEqual(result.shape, frame.shape)

    def test_output_dtype_uint8(self):
        frame = self._random_frame()
        mask = _butterworth_bandstop_mask(*frame.shape[:2])
        result = apply_fft_notch_to_frame(frame, mask)
        self.assertEqual(result.dtype, np.uint8)

    def test_all_black_frame_unchanged(self):
        """All-black frame has no frequency content — output must stay all-zero."""
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = _butterworth_bandstop_mask(64, 64)
        result = apply_fft_notch_to_frame(frame, mask)
        np.testing.assert_array_equal(result, frame)

    def test_all_white_frame_unchanged(self):
        """All-white frame is DC-only — the band-stop must not attenuate it."""
        frame = np.full((64, 64, 3), 255, dtype=np.uint8)
        mask = _butterworth_bandstop_mask(64, 64)
        result = apply_fft_notch_to_frame(frame, mask)
        # Allow ±1 rounding error from float64 → uint8 cast.
        np.testing.assert_array_almost_equal(
            result.astype(np.float32), frame.astype(np.float32), decimal=0
        )

    def test_values_clipped_to_uint8_range(self):
        """Output values must not exceed [0, 255] even for high-frequency inputs."""
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = _butterworth_bandstop_mask(64, 64)
        result = apply_fft_notch_to_frame(frame, mask)
        self.assertGreaterEqual(int(result.min()), 0)
        self.assertLessEqual(int(result.max()), 255)


# ---------------------------------------------------------------------------
# Container metadata strip
# ---------------------------------------------------------------------------


class TestStripContainerMetadata(unittest.TestCase):
    """strip_container_metadata fallback and success paths."""

    @patch("watermark_clean.subprocess.run")
    def test_returns_out_path_on_ffmpeg_success(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(returncode=0, stderr=b"")
        src, dst = Path("src.mp4"), Path("dst.mp4")
        self.assertEqual(strip_container_metadata(src, dst), dst)

    @patch("watermark_clean.subprocess.run")
    def test_returns_original_on_ffmpeg_failure(self, mock_run: MagicMock):
        """Non-zero ffmpeg exit must silently fall back to the input path."""
        mock_run.return_value = MagicMock(returncode=1, stderr=b"ffmpeg error")
        src, dst = Path("src.mp4"), Path("dst.mp4")
        self.assertEqual(strip_container_metadata(src, dst), src)

    @patch("watermark_clean.subprocess.run")
    def test_map_metadata_flag_present(self, mock_run: MagicMock):
        """-map_metadata -1 must always appear in the ffmpeg command."""
        mock_run.return_value = MagicMock(returncode=0, stderr=b"")
        strip_container_metadata(Path("a.mp4"), Path("b.mp4"))
        cmd: list[str] = mock_run.call_args[0][0]
        self.assertIn("-map_metadata", cmd)
        idx = cmd.index("-map_metadata")
        self.assertEqual(cmd[idx + 1], "-1")

    @patch("watermark_clean.subprocess.run")
    def test_map_chapters_flag_present(self, mock_run: MagicMock):
        """-map_chapters -1 must always appear in the ffmpeg command."""
        mock_run.return_value = MagicMock(returncode=0, stderr=b"")
        strip_container_metadata(Path("a.mp4"), Path("b.mp4"))
        cmd: list[str] = mock_run.call_args[0][0]
        self.assertIn("-map_chapters", cmd)

    @patch("watermark_clean.subprocess.run")
    def test_stream_copy_no_reencode(self, mock_run: MagicMock):
        """-c:v copy must be present so pixels are never re-encoded."""
        mock_run.return_value = MagicMock(returncode=0, stderr=b"")
        strip_container_metadata(Path("a.mp4"), Path("b.mp4"))
        cmd: list[str] = mock_run.call_args[0][0]
        self.assertIn("copy", cmd)


# ---------------------------------------------------------------------------
# deep_clean_frames dispatcher
# ---------------------------------------------------------------------------


class TestDeepCleanFrames(unittest.TestCase):
    """deep_clean_frames routing and error fallback."""

    @patch("watermark_clean.fft_clean")
    def test_fast_mode_calls_fft_clean(self, mock_fft: MagicMock):
        """mode='fast' must delegate to fft_clean."""
        expected = Path("out.mp4")
        mock_fft.return_value = expected
        result = deep_clean_frames(Path("in.mp4"), expected, "libx264", 10.0, mode="fast")
        mock_fft.assert_called_once()
        self.assertEqual(result, expected)

    @patch("watermark_clean.diffusion_clean")
    def test_paranoid_mode_calls_diffusion_clean(self, mock_diff: MagicMock):
        """mode='paranoid' must delegate to diffusion_clean."""
        expected = Path("out.mp4")
        mock_diff.return_value = expected
        result = deep_clean_frames(Path("in.mp4"), expected, "h264_nvenc", 10.0, mode="paranoid")
        mock_diff.assert_called_once()
        self.assertEqual(result, expected)

    @patch("watermark_clean.fft_clean", side_effect=RuntimeError("GPU OOM"))
    def test_fallback_on_exception(self, _mock: MagicMock):
        """Any RuntimeError from the backend must return the original path safely."""
        src = Path("in.mp4")
        result = deep_clean_frames(src, Path("out.mp4"), "libx264", 10.0, mode="fast")
        self.assertEqual(result, src)


if __name__ == "__main__":
    unittest.main()
