#!/usr/bin/env python3
"""
Extract GitHub repository URLs and code language information from extracted text files.

Thin orchestrator: all extraction logic lives in
infrastructure/literature/tool_analysis/repo_extractor.py.
"""

import argparse
import sys
from pathlib import Path

# Add root to path for infrastructure imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.literature.tool_analysis import repo_extractor


def main():
    parser = argparse.ArgumentParser(
        description="Extract GitHub repository URLs from extracted paper text."
    )
    parser.add_argument(
        "--extracted-text-dir", default="data/extracted_text",
        help="Directory of extracted paper text files (default: data/extracted_text)"
    )
    parser.add_argument(
        "--library-file", default="data/library.json",
        help="Library metadata JSON (default: data/library.json)"
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Directory for the github_repositories.md report (default: data)"
    )
    args = parser.parse_args()

    repo_extractor.run(
        extracted_text_dir=Path(args.extracted_text_dir),
        library_file=Path(args.library_file),
        output_dir=Path(args.output_dir),
    )


if __name__ == '__main__':
    main()
