"""Unit tests for PARANOID-mode diffusion parameter validation."""

import unittest

import numpy as np

from watermark_diffusion import _blend_with_source_frame, _effective_inference_steps


class TestEffectiveInferenceSteps(unittest.TestCase):
    """Verify Diffusers schedules work without raising the denoise strength."""

    def test_raises_scheduler_steps_for_low_denoise(self) -> None:
        """A low strength must use enough scheduler steps to run one pass."""
        self.assertEqual(_effective_inference_steps(0.10, 2), 10)

    def test_preserves_sufficient_scheduler_steps(self) -> None:
        """Existing steps remain unchanged when they already schedule work."""
        self.assertEqual(_effective_inference_steps(0.75, 2), 2)

    def test_rejects_invalid_strength(self) -> None:
        """Zero strength cannot produce an img2img denoising schedule."""
        with self.assertRaises(ValueError):
            _effective_inference_steps(0.0, 2)


class TestBlendWithSourceFrame(unittest.TestCase):
    """Verify source anchoring preserves intended blend boundaries."""

    def test_zero_weight_returns_source_frame(self) -> None:
        """No diffusion contribution must preserve the source exactly."""
        source = np.full((2, 2, 3), 30, dtype=np.uint8)
        diffused = np.full((2, 2, 3), 230, dtype=np.uint8)
        np.testing.assert_array_equal(
            _blend_with_source_frame(source, diffused, 0.0),
            source,
        )

    def test_full_weight_returns_diffusion_frame(self) -> None:
        """A full diffusion contribution must return the diffusion result."""
        source = np.full((2, 2, 3), 30, dtype=np.uint8)
        diffused = np.full((2, 2, 3), 230, dtype=np.uint8)
        np.testing.assert_array_equal(
            _blend_with_source_frame(source, diffused, 1.0),
            diffused,
        )

    def test_partial_weight_keeps_the_source_dominant(self) -> None:
        """The default-style blend must retain most of the source pixel value."""
        source = np.full((2, 2, 3), 30, dtype=np.uint8)
        diffused = np.full((2, 2, 3), 230, dtype=np.uint8)
        result = _blend_with_source_frame(source, diffused, 0.15)
        np.testing.assert_array_equal(result, np.full((2, 2, 3), 60, dtype=np.uint8))

    def test_resizes_diffusion_frame_before_blending(self) -> None:
        """A model-resized output must be restored to the source dimensions."""
        source = np.zeros((4, 6, 3), dtype=np.uint8)
        diffused = np.full((2, 3, 3), 200, dtype=np.uint8)
        result = _blend_with_source_frame(source, diffused, 1.0)
        self.assertEqual(result.shape, source.shape)
        np.testing.assert_array_equal(result, np.full(source.shape, 200, dtype=np.uint8))

    def test_rejects_mismatched_frame_channel_counts(self) -> None:
        """Frames with different channel counts cannot be blended safely."""
        source = np.zeros((2, 2, 3), dtype=np.uint8)
        diffused = np.zeros((2, 2, 1), dtype=np.uint8)
        with self.assertRaises(ValueError):
            _blend_with_source_frame(source, diffused, 0.15)
