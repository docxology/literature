#!/usr/bin/env python3
"""
Compare analysis runs to see what changed.
"""

import re
import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

# Pattern to find GitHub URLs
github_pattern = re.compile(r'https?://(?:www\.)?github\.com/[^\s\)]+|github\.com/[^\s\)]+', re.IGNORECASE)

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

def analyze_current_state():
    """Analyze current state of extracted text files."""
    results = {
        'total_files': 0,
        'files_with_github': 0,
        'files_with_year': 0,
        'files_analyzed': 0,
        'years': defaultdict(int),
        'papers_by_year': defaultdict(list)
    }
    
    extracted_text_dir = Path('data/extracted_text')
    
    if not extracted_text_dir.exists():
        return results
    
    for txt_file in sorted(extracted_text_dir.glob('*.txt')):
        results['total_files'] += 1
        
        try:
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check for GitHub URLs
            if github_pattern.search(content):
                results['files_with_github'] += 1
                paper_name = txt_file.stem
                year = extract_year_from_citation_key(paper_name)
                
                if year:
                    results['files_with_year'] += 1
                    results['files_analyzed'] += 1
                    results['years'][year] += 1
                    results['papers_by_year'][year].append(paper_name)
        except Exception as e:
            print(f'Error processing {txt_file}: {e}', file=__import__('sys').stderr)
    
    return results

def read_previous_summary():
    """Read previous summary if it exists."""
    summary_path = Path('data/plots/tools_and_languages_summary.md')
    if not summary_path.exists():
        return None
    
    try:
        with open(summary_path, 'r') as f:
            content = f.read()
        
        # Extract key stats
        stats = {}
        match = re.search(r'Total Papers Analyzed:\*\* (\d+)', content)
        if match:
            stats['papers'] = int(match.group(1))
        
        match = re.search(r'Year Range:\*\* (\d+) - (\d+)', content)
        if match:
            stats['year_min'] = int(match.group(1))
            stats['year_max'] = int(match.group(2))
        
        match = re.search(r'Total Unique Languages:\*\* (\d+)', content)
        if match:
            stats['languages'] = int(match.group(1))
        
        match = re.search(r'Total Unique Tools:\*\* (\d+)', content)
        if match:
            stats['tools'] = int(match.group(1))
        
        # Extract date
        match = re.search(r'\*\*Analysis Date:\*\* (\d{4}-\d{2}-\d{2})', content)
        if match:
            stats['date'] = match.group(1)
        
        return stats
    except Exception as e:
        print(f'Error reading previous summary: {e}')
        return None

def create_comparison_report(current, previous):
    """Create comparison report."""
    output_path = Path('data/plots/analysis_comparison.md')
    
    with open(output_path, 'w') as f:
        f.write("# Analysis Comparison Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## Current State\n\n")
        f.write(f"- **Total extracted text files:** {current['total_files']}\n")
        f.write(f"- **Files with GitHub URLs:** {current['files_with_github']}\n")
        f.write(f"- **Files with valid years:** {current['files_with_year']}\n")
        f.write(f"- **Files analyzed:** {current['files_analyzed']}\n")
        f.write(f"- **Year range:** {min(current['years'].keys()) if current['years'] else 'N/A'} - {max(current['years'].keys()) if current['years'] else 'N/A'}\n\n")
        
        if previous:
            f.write("## Previous Analysis\n\n")
            f.write(f"- **Papers analyzed:** {previous.get('papers', 'N/A')}\n")
            f.write(f"- **Year range:** {previous.get('year_min', 'N/A')} - {previous.get('year_max', 'N/A')}\n")
            f.write(f"- **Languages:** {previous.get('languages', 'N/A')}\n")
            f.write(f"- **Tools:** {previous.get('tools', 'N/A')}\n")
            f.write(f"- **Analysis date:** {previous.get('date', 'N/A')}\n\n")
            
            f.write("## Changes\n\n")
            papers_diff = current['files_analyzed'] - previous.get('papers', 0)
            if papers_diff > 0:
                f.write(f"✅ **+{papers_diff} new papers** analyzed\n")
            elif papers_diff < 0:
                f.write(f"⚠️ **{papers_diff} papers** (possible data issue)\n")
            else:
                f.write("➡️ **No change** in number of papers\n")
        else:
            f.write("## Previous Analysis\n\n")
            f.write("No previous analysis found.\n\n")
        
        f.write("\n## Papers by Year (Current)\n\n")
        f.write("| Year | Count | Papers |\n")
        f.write("|------|-------|--------|\n")
        for year in sorted(current['years'].keys()):
            count = current['years'][year]
            papers = current['papers_by_year'][year][:5]  # Show first 5
            papers_str = ', '.join(papers)
            if len(current['papers_by_year'][year]) > 5:
                papers_str += f", ... ({len(current['papers_by_year'][year]) - 5} more)"
            f.write(f"| {year} | {count} | {papers_str} |\n")
        
        # Show papers without years
        f.write("\n## Papers Missing Years\n\n")
        missing_year_count = current['files_with_github'] - current['files_with_year']
        if missing_year_count > 0:
            f.write(f"**{missing_year_count} papers** have GitHub URLs but couldn't extract year from citation key.\n")
            f.write("\nThese papers are excluded from time-series analysis.\n")
        else:
            f.write("All papers with GitHub URLs have valid years.\n")
    
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    print("Analyzing current state...")
    current = analyze_current_state()
    
    print("\nReading previous analysis...")
    previous = read_previous_summary()
    
    print("\nCreating comparison report...")
    create_comparison_report(current, previous)
    
    print("\nSummary:")
    print(f"  Total files: {current['total_files']}")
    print(f"  Files with GitHub: {current['files_with_github']}")
    print(f"  Files analyzed: {current['files_analyzed']}")
    if previous:
        diff = current['files_analyzed'] - previous.get('papers', 0)
        if diff != 0:
            print(f"  Change: {diff:+d} papers")
        else:
            print(f"  Change: No change")
