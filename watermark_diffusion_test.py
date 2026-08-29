"""Unit tests for PARANOID-mode diffusion parameter validation."""

import unittest

from watermark_diffusion import _effective_denoise_strength


class TestEffectiveDenoiseStrength(unittest.TestCase):
    """Verify Diffusers always receives at least one denoising timestep."""

    def test_raises_strength_to_one_step_minimum(self) -> None:
        """A low strength with two steps must still run one denoising step."""
        self.assertEqual(_effective_denoise_strength(0.10, 2), 0.5)

    def test_preserves_strength_that_already_runs(self) -> None:
        """A strength that schedules work is not changed."""
        self.assertEqual(_effective_denoise_strength(0.75, 2), 0.75)

    def test_rejects_invalid_strength(self) -> None:
        """Zero strength cannot produce an img2img denoising schedule."""
        with self.assertRaises(ValueError):
            _effective_denoise_strength(0.0, 2)

