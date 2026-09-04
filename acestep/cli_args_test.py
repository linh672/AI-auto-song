"""Unit tests for CLI argument parsing helpers."""

from __future__ import annotations

import argparse
import unittest

from acestep.cli_args import normalize_batch_args, parse_quantization_arg


class NormalizeBatchArgsTests(unittest.TestCase):
    """Behavior tests for batch argument normalization."""

    def test_normalizes_double_dash_batch_value(self) -> None:
        """--batch --100 is normalized to --batch 100."""
        argv = ["acestep", "--batch", "--100"]
        self.assertEqual(["acestep", "--batch", "100"], normalize_batch_args(argv))

    def test_normalizes_equals_double_dash_batch_value(self) -> None:
        """--batch=--100 is normalized to --batch=100."""
        argv = ["acestep", "--batch=--100"]
        self.assertEqual(["acestep", "--batch=100"], normalize_batch_args(argv))

    def test_normalizes_standalone_double_dash_number(self) -> None:
        """--100 is normalized to --batch 100."""
        argv = ["acestep", "--100"]
        self.assertEqual(["acestep", "--batch", "100"], normalize_batch_args(argv))

    def test_preserves_plain_batch_flag(self) -> None:
        """--batch without number is preserved for const default."""
        argv = ["acestep", "--batch"]
        self.assertEqual(["acestep", "--batch"], normalize_batch_args(argv))

    def test_preserves_batch_with_standard_number(self) -> None:
        """--batch 50 is preserved."""
        argv = ["acestep", "--batch", "50"]
        self.assertEqual(["acestep", "--batch", "50"], normalize_batch_args(argv))

    def test_handles_batch_followed_by_another_flag(self) -> None:
        """--batch --port 7860 leaves --port untouched."""
        argv = ["acestep", "--batch", "--port", "7860"]
        self.assertEqual(["acestep", "--batch", "--port", "7860"], normalize_batch_args(argv))

    def test_preserves_negative_option_values(self) -> None:
        """-1 (single dash) is not converted to --batch."""
        argv = ["acestep", "--audio_duration", "-1"]
        self.assertEqual(["acestep", "--audio_duration", "-1"], normalize_batch_args(argv))

    def test_empty_argv(self) -> None:
        """Empty argv returns empty list."""
        self.assertEqual([], normalize_batch_args([]))


class ParseQuantizationArgTests(unittest.TestCase):
    """Behavior tests for quantization CLI parsing."""

    def test_returns_none_for_none_aliases(self) -> None:
        """It treats ``none`` aliases as disabled quantization."""
        self.assertIsNone(parse_quantization_arg("none"))
        self.assertIsNone(parse_quantization_arg("None"))
        self.assertIsNone(parse_quantization_arg(" null "))
        self.assertIsNone(parse_quantization_arg(""))

    def test_returns_canonical_quantization_values(self) -> None:
        """It returns supported quantization values in canonical form."""
        self.assertEqual("int8_weight_only", parse_quantization_arg("int8_weight_only"))
        self.assertEqual("int8_weight_only", parse_quantization_arg("INT8_WEIGHT_ONLY"))
        self.assertEqual("fp8_weight_only", parse_quantization_arg("fp8_weight_only"))
        self.assertEqual("fp8_weight_only", parse_quantization_arg(" FP8_Weight_Only "))
        self.assertEqual("w8a8_dynamic", parse_quantization_arg("w8a8_dynamic"))
        self.assertEqual("w8a8_dynamic", parse_quantization_arg(" W8A8_Dynamic "))

    def test_raises_for_invalid_value(self) -> None:
        """It raises ``ArgumentTypeError`` for unsupported values."""
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_quantization_arg("int4_weight_only")


if __name__ == "__main__":
    unittest.main()

