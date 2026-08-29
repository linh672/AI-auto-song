"""Unit tests for PARANOID-mode diffusion parameter validation."""

import unittest

from watermark_diffusion import _effective_inference_steps


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
