"""Entry point: generate input videos via Google Flow automation.

Usage:
    python create_input.py
    python create_input.py --dry-run
    python create_input.py --cooldown 30
    python create_input.py --headless
"""

import sys

from automation.run_pipeline import run_pipeline

if __name__ == "__main__":
    sys.exit(run_pipeline())
