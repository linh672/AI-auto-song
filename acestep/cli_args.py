"""Argument parsing helpers shared by ACE-Step CLI entrypoints."""

from __future__ import annotations

import argparse

_QUANTIZATION_ALIASES = {
    "int8_weight_only": "int8_weight_only",
    "fp8_weight_only": "fp8_weight_only",
    "w8a8_dynamic": "w8a8_dynamic",
}
_NONE_ALIASES = {"", "none", "null"}


def parse_quantization_arg(value: str | None) -> str | None:
    """Parse ``--quantization`` values from CLI input.

    Args:
        value: Raw CLI value.

    Returns:
        Canonical quantization method or ``None`` for disabled quantization.

    Raises:
        argparse.ArgumentTypeError: If the value is not supported.
    """
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in _NONE_ALIASES:
        return None

    quantization = _QUANTIZATION_ALIASES.get(normalized)
    if quantization is not None:
        return quantization

    raise argparse.ArgumentTypeError(
        "Invalid quantization value. Use int8_weight_only, fp8_weight_only, "
        "w8a8_dynamic, or none."
    )


def normalize_batch_args(argv: list[str] | None = None) -> list[str]:
    """Normalize CLI arguments so batch counts are parsed cleanly.

    Handles forms like:
    - ``--batch --100`` -> ``--batch 100``
    - ``--batch=--100`` -> ``--batch=100``
    - ``--100`` (standalone) -> ``--batch 100``
    - ``--batch`` -> unchanged (defaults to 100 via argparse const)
    - ``--batch 100`` -> unchanged

    Args:
        argv: Argument list (defaults to sys.argv).

    Returns:
        Normalized argument list.
    """
    import sys

    if argv is None:
        argv = sys.argv

    if not argv:
        return []

    prog = argv[0]
    args = argv[1:]
    normalized: list[str] = []

    i = 0
    while i < len(args):
        arg = args[i]
        # Handle --batch=--100 or --batch=-100
        if arg.startswith("--batch="):
            val = arg.split("=", 1)[1]
            clean_val = val.lstrip("-")
            if clean_val.isdigit():
                normalized.append(f"--batch={clean_val}")
            else:
                normalized.append(arg)
            i += 1
            continue

        # Handle --batch followed by --<digits> or -<digits>
        if arg == "--batch":
            normalized.append(arg)
            if i + 1 < len(args):
                next_arg = args[i + 1]
                clean_next = next_arg.lstrip("-")
                if next_arg.startswith("-") and clean_next.isdigit():
                    normalized.append(clean_next)
                    i += 2
                    continue
            i += 1
            continue

        # Handle standalone --<digits> (e.g. --100)
        clean_arg = arg.lstrip("-")
        if arg.startswith("--") and clean_arg.isdigit():
            normalized.extend(["--batch", clean_arg])
            i += 1
            continue

        normalized.append(arg)
        i += 1

    return [prog, *normalized]

