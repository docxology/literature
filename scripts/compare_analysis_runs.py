#!/usr/bin/env python3
"""
Compare analysis runs to see what changed.

Thin orchestrator: all comparison logic lives in
infrastructure/literature/tool_analysis/run_comparison.py.
"""

import argparse
import sys
from pathlib import Path

# Add root to path for infrastructure imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.literature.tool_analysis import run_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Compare the current extraction state against the previous analysis run."
    )
    parser.add_argument(
        "--extracted-text-dir", default="data/extracted_text",
        help="Directory of extracted paper text files (default: data/extracted_text)"
    )
    parser.add_argument(
        "--output-dir", default="data/plots",
        help="Directory holding tools_and_languages_summary.md and the comparison report (default: data/plots)"
    )
    args = parser.parse_args()

    run_comparison.run(
        extracted_text_dir=Path(args.extracted_text_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == '__main__':
    main()
