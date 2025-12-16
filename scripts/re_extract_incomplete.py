#!/usr/bin/env python3
"""Script to re-extract incomplete HTML text files."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from infrastructure.literature.pdf.re_extract import check_and_re_extract_incomplete_files
from infrastructure.literature.core.config import LiteratureConfig

def main():
    """Main entry point."""
    library_path = Path("data/library.json")
    pdf_dir = Path("data/pdfs")
    config = LiteratureConfig.from_env()
    
    print("Checking and re-extracting incomplete files...")
    print(f"Minimum word count: 1000")
    print()
    
    results = check_and_re_extract_incomplete_files(
        library_path=library_path,
        pdf_dir=pdf_dir,
        min_words=1000,
        config=config
    )
    
    # Print summary
    re_extracted = [key for key, status in results if status == 're-extracted']
    failed = [key for key, status in results if status == 'failed']
    
    if re_extracted:
        print(f"\n✓ Successfully re-extracted {len(re_extracted)} files:")
        for key in re_extracted:
            print(f"  - {key}")
    
    if failed:
        print(f"\n✗ Failed to re-extract {len(failed)} files:")
        for key in failed:
            print(f"  - {key}")

if __name__ == "__main__":
    main()



