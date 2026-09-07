"""Extract GitHub repository URLs and code language information from extracted text.

Moved from scripts/extract_github_repos.py.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from infrastructure.literature.tool_analysis.common import (
    DEFAULT_EXTRACTED_TEXT_DIR,
    DEFAULT_LIBRARY_FILE,
    find_code_mentions,
    find_github_urls,
)


def load_library_metadata(library_file=DEFAULT_LIBRARY_FILE):
    """Load paper metadata from library.json."""
    library_file = Path(library_file)
    metadata = {}

    if library_file.exists():
        try:
            with open(library_file, 'r', encoding='utf-8') as f:
                library_data = json.load(f)
                entries = library_data.get('entries', {})
                for citation_key, entry in entries.items():
                    metadata[citation_key] = {
                        'title': entry.get('title', ''),
                        'authors': entry.get('authors', []),
                        'year': entry.get('year'),
                        'venue': entry.get('venue'),
                        'doi': entry.get('doi'),
                    }
        except Exception as e:
            print(f'Warning: Could not load library.json: {e}', file=sys.stderr)

    return metadata


def extract_paper_info_from_text(content, citation_key):
    """Extract paper title and authors from extracted text."""
    lines = content.split('\n')
    title = ''
    authors = []
    year = None

    # Try to find title in first 50 lines (usually near the beginning)
    for i, line in enumerate(lines[:50]):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Title is usually a longer line, not too short, not too long
        # Often appears before authors
        if 20 < len(line_stripped) < 200 and not title:
            # Check if it looks like a title (capitalized, not all caps, not a URL)
            if (line_stripped[0].isupper() and
                not line_stripped.isupper() and
                'http' not in line_stripped.lower() and
                'github' not in line_stripped.lower() and
                not line_stripped.startswith('DOI:') and
                not line_stripped.startswith('Abstract')):
                title = line_stripped
                # Don't break, might find better title

        # Look for author patterns (lines with "and" or commas, or "et al")
        if (' and ' in line_stripped.lower() or
            (',' in line_stripped and len(line_stripped.split(',')) <= 5) or
            'et al' in line_stripped.lower()):
            # Extract potential author names
            if len(line_stripped) < 300:  # Reasonable author line length
                # Try to parse authors
                if ' and ' in line_stripped:
                    parts = line_stripped.split(' and ')
                    authors.extend([p.strip() for p in parts if p.strip()])
                elif ',' in line_stripped:
                    parts = line_stripped.split(',')
                    authors.extend([p.strip() for p in parts[:5] if p.strip()])
                else:
                    authors.append(line_stripped)

    # Extract year from citation key if present
    year_match = re.search(r'(\d{4})', citation_key)
    if year_match:
        try:
            year = int(year_match.group(1))
        except:
            pass

    # Clean up title (take first reasonable candidate)
    if title:
        # Remove common prefixes
        title = re.sub(r'^(Title|TITLE|Paper|PAPER):\s*', '', title, flags=re.IGNORECASE)
        title = title.strip()

    # Clean up authors
    if authors:
        # Remove common prefixes
        authors_clean = []
        for author in authors[:5]:  # Limit to first 5
            author = re.sub(r'^(Authors?|AUTHORS?|By):\s*', '', author, flags=re.IGNORECASE)
            author = author.strip()
            if author and len(author) > 2:
                authors_clean.append(author)
        authors = authors_clean

    return {
        'title': title[:200] if title else '',  # Limit title length
        'authors': authors[:5] if authors else [],  # Limit to 5 authors
        'year': year,
    }


def extract_github_info(extracted_text_dir=DEFAULT_EXTRACTED_TEXT_DIR, library_file=DEFAULT_LIBRARY_FILE):
    """Extract GitHub URLs and code information from all extracted text files."""
    results = []
    extracted_text_dir = Path(extracted_text_dir)
    library_metadata = load_library_metadata(library_file)

    if not extracted_text_dir.exists():
        print(f"Error: Directory {extracted_text_dir} does not exist")
        return []

    for txt_file in sorted(extracted_text_dir.glob('*.txt')):
        try:
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')

            # Find GitHub URLs
            cleaned_urls = find_github_urls(content)

            if cleaned_urls:
                # Find code mentions
                code_mentions = find_code_mentions(content, cleaned_urls, lines)

                paper_name = txt_file.stem

                # Get paper metadata from library first, then try extracting from text
                paper_info = library_metadata.get(paper_name, {})

                # If no library metadata, try extracting from text
                if not paper_info.get('title'):
                    paper_info.update(extract_paper_info_from_text(content, paper_name))

                results.append({
                    'paper': paper_name,
                    'title': paper_info.get('title', ''),
                    'authors': paper_info.get('authors', []),
                    'year': paper_info.get('year'),
                    'venue': paper_info.get('venue'),
                    'doi': paper_info.get('doi'),
                    'urls': cleaned_urls,
                    'languages': sorted(code_mentions) if code_mentions else ['Not specified']
                })
        except Exception as e:
            print(f'Error processing {txt_file}: {e}', file=sys.stderr)

    return results


def generate_markdown_report(results):
    """Generate a markdown report of all GitHub repositories."""
    output = []
    output.append("# GitHub Repositories Found in Literature\n")
    output.append(f"This document contains all GitHub repository URLs found in the extracted text files.\n")
    output.append(f"**Total papers with GitHub repositories: {len(results)}**\n")
    output.append("---\n\n")

    # Group by language if possible
    by_language = defaultdict(list)
    for result in results:
        lang_key = ', '.join(result['languages']) if result['languages'] != ['Not specified'] else 'Not specified'
        by_language[lang_key].append(result)

    # Sort languages
    sorted_langs = sorted(by_language.keys(), key=lambda x: (x == 'Not specified', x))

    for lang in sorted_langs:
        if lang != 'Not specified':
            output.append(f"## {lang}\n\n")
        else:
            output.append("## Not Specified\n\n")

        papers = sorted(by_language[lang], key=lambda x: x['paper'])
        for result in papers:
            # Paper header with citation key
            output.append(f"### {result['paper']}\n\n")

            # Paper title and metadata
            if result.get('title'):
                output.append(f"**Paper:** {result['title']}\n\n")
            if result.get('authors'):
                authors_str = ', '.join(result['authors'][:3])
                if len(result['authors']) > 3:
                    authors_str += f" et al. ({len(result['authors'])} authors)"
                output.append(f"**Authors:** {authors_str}\n\n")
            if result.get('year'):
                output.append(f"**Year:** {result['year']}\n\n")
            if result.get('venue'):
                output.append(f"**Venue:** {result['venue']}\n\n")

            # GitHub repositories
            output.append("**Repositories:**\n")
            for url in result['urls']:
                output.append(f"- {url}\n")

            if result['languages'] != ['Not specified']:
                output.append(f"\n**Languages/Tools:** {', '.join(result['languages'])}\n")
            output.append("\n")

    # Also create a flat list
    output.append("\n---\n\n")
    output.append("## Complete List (Alphabetical by Paper)\n\n")

    for result in sorted(results, key=lambda x: x['paper']):
        # Paper header with citation key
        output.append(f"### {result['paper']}\n\n")

        # Paper title and metadata
        if result.get('title'):
            output.append(f"**Paper:** {result['title']}\n\n")
        if result.get('authors'):
            authors_str = ', '.join(result['authors'][:3])
            if len(result['authors']) > 3:
                authors_str += f" et al. ({len(result['authors'])} authors)"
            output.append(f"**Authors:** {authors_str}\n\n")
        if result.get('year'):
            output.append(f"**Year:** {result['year']}\n\n")
        if result.get('venue'):
            output.append(f"**Venue:** {result['venue']}\n\n")

        # GitHub repositories
        output.append("**Repositories:**\n")
        for url in result['urls']:
            output.append(f"- {url}\n")

        output.append(f"**Languages/Tools:** {', '.join(result['languages'])}\n\n")

    return ''.join(output)


def run(output_dir=Path('data'), extracted_text_dir=DEFAULT_EXTRACTED_TEXT_DIR, library_file=DEFAULT_LIBRARY_FILE):
    """Run the GitHub repository extraction and report generation."""
    output_dir = Path(output_dir)

    print("Extracting GitHub repository information...")
    results = extract_github_info(extracted_text_dir, library_file)

    print(f"\nFound {len(results)} papers with GitHub repositories")

    # Generate markdown report
    report = generate_markdown_report(results)

    output_file = output_dir / 'github_repositories.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to: {output_file}")
    print(f"\nSummary:")
    for result in sorted(results, key=lambda x: x['paper'])[:10]:  # Show first 10
        print(f"  {result['paper']}: {len(result['urls'])} repo(s) - {', '.join(result['languages'])}")
    if len(results) > 10:
        print(f"  ... and {len(results) - 10} more")
