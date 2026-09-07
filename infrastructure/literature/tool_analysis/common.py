"""Shared extraction utilities for repository/tool analysis.

Single home for the GitHub-URL pattern, URL cleaning, citation-key year
extraction, language/tool pattern tables, and context-window mention search.
Moved verbatim from the former inline copies in scripts/
analyze_specific_tools.py, analyze_tool_cooccurrence.py,
analyze_tools_and_languages_over_time.py, plot_language_trends.py,
extract_github_repos.py, and compare_analysis_runs.py.
"""

import re
from pathlib import Path

# Pattern to find GitHub URLs
github_pattern = re.compile(r'https?://(?:www\.)?github\.com/[^\s\)]+|github\.com/[^\s\)]+', re.IGNORECASE)

# Pattern to find programming languages/technologies
code_patterns = {
    'Python': re.compile(r'\bpython\b', re.IGNORECASE),
    'MATLAB': re.compile(r'\bmatlab\b', re.IGNORECASE),
    'Julia': re.compile(r'\bjulia\b', re.IGNORECASE),
    'R': re.compile(r'\b\bR\b\b', re.IGNORECASE),
    'C++': re.compile(r'\bc\+\+\b', re.IGNORECASE),
    'Java': re.compile(r'\bjava\b', re.IGNORECASE),
    'JavaScript': re.compile(r'\bjavascript\b', re.IGNORECASE),
    'TypeScript': re.compile(r'\btypescript\b', re.IGNORECASE),
    'PyTorch': re.compile(r'\bpytorch\b', re.IGNORECASE),
    'TensorFlow': re.compile(r'\btensorflow\b', re.IGNORECASE),
    'JAX': re.compile(r'\bjax\b', re.IGNORECASE),
    'ROS': re.compile(r'\bros\b', re.IGNORECASE),
    'ROS2': re.compile(r'\bros2\b', re.IGNORECASE),
}

# Programming languages (subset of code_patterns used by the time-series analysis)
languages = {
    'Python': re.compile(r'\bpython\b', re.IGNORECASE),
    'MATLAB': re.compile(r'\bmatlab\b', re.IGNORECASE),
    'Julia': re.compile(r'\bjulia\b', re.IGNORECASE),
    'R': re.compile(r'\b\bR\b\b', re.IGNORECASE),
    'C++': re.compile(r'\bc\+\+\b', re.IGNORECASE),
    'Java': re.compile(r'\bjava\b', re.IGNORECASE),
    'JavaScript': re.compile(r'\bjavascript\b', re.IGNORECASE),
    'TypeScript': re.compile(r'\btypescript\b', re.IGNORECASE),
}

# Default locations relative to the repository root (script working directory)
DEFAULT_EXTRACTED_TEXT_DIR = Path('data/extracted_text')
DEFAULT_PLOTS_DIR = Path('data/plots')
DEFAULT_LIBRARY_FILE = Path('data/library.json')


def clean_url(url):
    """Clean and normalize GitHub URL."""
    # Remove trailing punctuation and common suffixes
    url = url.rstrip('.,;:!?)')
    # Remove trailing brackets and other characters
    url = url.rstrip(']')
    # Remove trailing year references like ",2019" or ",2020"
    url = re.sub(r',\s*\d{4}$', '', url)
    # Remove trailing version numbers like ",3" or ",2.0"
    url = re.sub(r',\s*\d+(?:\.\d+)?$', '', url)
    # Remove trailing dashes (incomplete URLs)
    url = url.rstrip('-')
    # Remove query parameters and fragments (like ;visit=...)
    if ';' in url:
        url = url.split(';')[0]
    # Remove trailing paths that are incomplete (ending with /)
    if url.endswith('/') and url.count('/') > 4:
        url = url.rstrip('/')
    # Skip URLs that are clearly incomplete (too short or end with special chars)
    if len(url.replace('https://github.com/', '').replace('http://github.com/', '')) < 3:
        return None
    if url.endswith('-') or url.endswith(']') or url.endswith('['):
        return None
    # Add https:// if missing
    if not url.startswith('http'):
        url = 'https://' + url
    return url


def extract_year_from_citation_key(citation_key):
    """Extract year from citation key."""
    year_match = re.search(r'(\d{4})', citation_key)
    if year_match:
        try:
            year = int(year_match.group(1))
            if 1990 <= year <= 2030:
                return year
        except:
            pass
    return None


def find_github_urls(content):
    """Find GitHub URLs in content: match, clean, and de-duplicate preserving order."""
    cleaned_urls = []
    for url in github_pattern.findall(content):
        cleaned = clean_url(url)
        if cleaned and cleaned not in cleaned_urls:
            cleaned_urls.append(cleaned)
    return cleaned_urls


def find_items_in_content(content, patterns, github_urls, lines, context_lines=15):
    """Find items matching patterns in content.

    Checks the full content plus the context window (``context_lines`` lines on
    either side) around every line that mentions one of the GitHub URLs.
    """
    items_found = set()
    content_lower = content.lower()

    # Check full content
    for item_name, pattern in patterns.items():
        if pattern.search(content_lower):
            items_found.add(item_name)

    # Also check context around GitHub URLs
    for i, line in enumerate(lines):
        if any(url.replace('https://', '').replace('http://', '') in line.lower()
               for url in github_urls):
            context_start = max(0, i - context_lines)
            context_end = min(len(lines), i + context_lines)
            context = ' '.join(lines[context_start:context_end]).lower()

            for item_name, pattern in patterns.items():
                if pattern.search(context):
                    items_found.add(item_name)

    return items_found


def find_code_mentions(content, github_urls, lines):
    """Find programming language mentions near GitHub URLs (10-line context window)."""
    return find_items_in_content(content, code_patterns, github_urls, lines, context_lines=10)
