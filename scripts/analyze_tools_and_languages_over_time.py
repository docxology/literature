#!/usr/bin/env python3
"""
Comprehensive analysis of tools and languages over time.
Creates a single summary document with trends for both.

Thin orchestrator: all analysis logic lives in
infrastructure/literature/tool_analysis/over_time.py.
"""

import argparse
import sys
from pathlib import Path

# Add root to path for infrastructure imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.literature.tool_analysis import over_time


def main():
    parser = argparse.ArgumentParser(
        description="Analyze tools and languages over time from extracted paper text."
    )
    parser.add_argument(
        "--extracted-text-dir", default="data/extracted_text",
        help="Directory of extracted paper text files (default: data/extracted_text)"
    )
    parser.add_argument(
        "--output-dir", default="data/plots",
        help="Directory for analysis outputs (default: data/plots)"
    )
    args = parser.parse_args()

    over_time.run(
        extracted_text_dir=Path(args.extracted_text_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == '__main__':
    main()
