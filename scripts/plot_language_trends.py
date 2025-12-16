#!/usr/bin/env python3
"""
Generate plots showing programming language usage trends over time from GitHub repositories.
Creates HTML visualizations using Chart.js and CSV data files.
"""

import json
import re
import csv
from pathlib import Path
from collections import defaultdict

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

def clean_url(url):
    """Clean and normalize GitHub URL."""
    url = url.rstrip('.,;:!?)')
    url = url.rstrip(']')
    url = re.sub(r',\s*\d{4}$', '', url)
    url = re.sub(r',\s*\d+(?:\.\d+)?$', '', url)
    url = url.rstrip('-')
    if ';' in url:
        url = url.split(';')[0]
    if url.endswith('/') and url.count('/') > 4:
        url = url.rstrip('/')
    if len(url.replace('https://github.com/', '').replace('http://github.com/', '')) < 3:
        return None
    if url.endswith('-') or url.endswith(']') or url.endswith('['):
        return None
    if not url.startswith('http'):
        url = 'https://' + url
    return url

def find_code_mentions(content, github_urls, lines):
    """Find programming language mentions near GitHub URLs."""
    code_mentions = set()
    
    for i, line in enumerate(lines):
        if any(url.replace('https://', '').replace('http://', '') in line.lower() 
               for url in github_urls):
            context_start = max(0, i-10)
            context_end = min(len(lines), i+10)
            context = ' '.join(lines[context_start:context_end]).lower()
            
            for lang, pattern in code_patterns.items():
                if pattern.search(context):
                    code_mentions.add(lang)
    
    content_lower = content.lower()
    for lang, pattern in code_patterns.items():
        if pattern.search(content_lower):
            code_mentions.add(lang)
    
    return code_mentions

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

def extract_data():
    """Extract GitHub repository data with years and languages."""
    results = []
    extracted_text_dir = Path('data/extracted_text')
    
    if not extracted_text_dir.exists():
        print(f"Error: Directory {extracted_text_dir} does not exist")
        return []
    
    for txt_file in sorted(extracted_text_dir.glob('*.txt')):
        try:
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            github_urls_raw = github_pattern.findall(content)
            
            if github_urls_raw:
                cleaned_urls = []
                for url in github_urls_raw:
                    cleaned = clean_url(url)
                    if cleaned and cleaned not in cleaned_urls:
                        cleaned_urls.append(cleaned)
                
                code_mentions = find_code_mentions(content, cleaned_urls, lines)
                paper_name = txt_file.stem
                year = extract_year_from_citation_key(paper_name)
                
                if year and code_mentions:
                    results.append({
                        'year': year,
                        'languages': code_mentions,
                        'paper': paper_name,
                        'repo_count': len(cleaned_urls)
                    })
        except Exception as e:
            print(f'Error processing {txt_file}: {e}', file=__import__('sys').stderr)
    
    return results

def prepare_chart_data(data):
    """Prepare data for charting."""
    year_lang_counts = defaultdict(lambda: defaultdict(int))
    
    for entry in data:
        year = entry['year']
        for lang in entry['languages']:
            year_lang_counts[year][lang] += 1
    
    years = sorted(year_lang_counts.keys())
    all_languages = set()
    for lang_counts in year_lang_counts.values():
        all_languages.update(lang_counts.keys())
    
    lang_data = {lang: [] for lang in all_languages}
    for year in years:
        for lang in all_languages:
            count = year_lang_counts[year].get(lang, 0)
            lang_data[lang].append(count)
    
    lang_totals = {lang: sum(lang_data[lang]) for lang in all_languages}
    sorted_langs = sorted(all_languages, key=lambda x: lang_totals[x], reverse=True)
    
    return years, lang_data, sorted_langs, year_lang_counts

def create_csv_data(data, output_dir):
    """Create CSV files with language usage data."""
    years, lang_data, sorted_langs, year_lang_counts = prepare_chart_data(data)
    
    # CSV 1: Language usage by year (wide format)
    csv_path = output_dir / 'language_usage_by_year.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Language'] + [str(y) for y in years])
        for lang in sorted_langs:
            writer.writerow([lang] + lang_data[lang])
    print(f"Saved: {csv_path}")
    
    # CSV 2: Long format (for easier plotting in some tools)
    csv_path_long = output_dir / 'language_usage_long.csv'
    with open(csv_path_long, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Year', 'Language', 'Count'])
        for year in years:
            for lang in sorted_langs:
                count = year_lang_counts[year].get(lang, 0)
                if count > 0:
                    writer.writerow([year, lang, count])
    print(f"Saved: {csv_path_long}")

def create_html_visualization(data, output_dir):
    """Create HTML file with interactive charts using Chart.js."""
    years, lang_data, sorted_langs, year_lang_counts = prepare_chart_data(data)
    
    # Get top languages for display
    top_langs = sorted_langs[:12]
    
    # Prepare data for Chart.js
    years_js = [str(y) for y in years]
    
    # Color palette
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78'
    ]
    
    # Prepare datasets for line chart
    datasets_line = []
    for i, lang in enumerate(top_langs):
        datasets_line.append({
            'label': lang,
            'data': lang_data[lang],
            'borderColor': colors[i % len(colors)],
            'backgroundColor': colors[i % len(colors)] + '40',
            'tension': 0.1
        })
    
    # Prepare datasets for stacked area chart
    datasets_stacked = []
    for i, lang in enumerate(top_langs[:8]):  # Top 8 for stacked
        datasets_stacked.append({
            'label': lang,
            'data': lang_data[lang],
            'backgroundColor': colors[i % len(colors)] + '80',
        })
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Programming Language Usage Trends</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
        canvas {{
            max-height: 500px;
        }}
        .stats {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .stats h2 {{
            margin-top: 0;
            color: #333;
        }}
        .stats p {{
            margin: 5px 0;
            color: #666;
        }}
    </style>
</head>
<body>
    <h1>Programming Language Usage Trends</h1>
    <p class="subtitle">GitHub Repositories in Academic Literature ({min(years)} - {max(years)})</p>
    
    <div class="stats">
        <h2>Summary Statistics</h2>
        <p><strong>Total papers with GitHub repositories:</strong> {len(data)}</p>
        <p><strong>Year range:</strong> {min(years)} - {max(years)}</p>
        <p><strong>Total unique languages/tools:</strong> {len(sorted_langs)}</p>
        <p><strong>Top languages:</strong> {', '.join(sorted_langs[:10])}</p>
    </div>
    
    <div class="chart-container">
        <div class="chart-title">Language Usage Trends Over Time (Line Chart)</div>
        <canvas id="lineChart"></canvas>
    </div>
    
    <div class="chart-container">
        <div class="chart-title">Language Usage Over Time (Stacked Area Chart)</div>
        <canvas id="stackedChart"></canvas>
    </div>
    
    <div class="chart-container">
        <div class="chart-title">Language Usage by Year (Bar Chart)</div>
        <canvas id="barChart"></canvas>
    </div>
    
    <script>
        const years = {json.dumps(years_js)};
        const datasetsLine = {json.dumps(datasets_line)};
        const datasetsStacked = {json.dumps(datasets_stacked)};
        
        const chartOptions = {{
            responsive: true,
            maintainAspectRatio: true,
            plugins: {{
                legend: {{
                    position: 'right',
                    labels: {{
                        boxWidth: 12,
                        padding: 10
                    }}
                }},
                tooltip: {{
                    mode: 'index',
                    intersect: false
                }}
            }},
            scales: {{
                x: {{
                    title: {{
                        display: true,
                        text: 'Year'
                    }}
                }},
                y: {{
                    title: {{
                        display: true,
                        text: 'Number of Papers'
                    }},
                    beginAtZero: true
                }}
            }}
        }};
        
        // Line Chart
        new Chart(document.getElementById('lineChart'), {{
            type: 'line',
            data: {{
                labels: years,
                datasets: datasetsLine
            }},
            options: chartOptions
        }});
        
        // Stacked Area Chart
        new Chart(document.getElementById('stackedChart'), {{
            type: 'line',
            data: {{
                labels: years,
                datasets: datasetsStacked
            }},
            options: {{
                ...chartOptions,
                plugins: {{
                    ...chartOptions.plugins,
                    tooltip: {{
                        mode: 'index',
                        intersect: false
                    }}
                }}
            }},
            scales: {{
                ...chartOptions.scales,
                y: {{
                    ...chartOptions.scales.y,
                    stacked: true
                }}
            }}
        }});
        
        // Bar Chart
        new Chart(document.getElementById('barChart'), {{
            type: 'bar',
            data: {{
                labels: years,
                datasets: datasetsLine
            }},
            options: {{
                ...chartOptions,
                scales: {{
                    ...chartOptions.scales,
                    y: {{
                        ...chartOptions.scales.y,
                        stacked: false
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
    
    output_path = output_dir / 'language_trends.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Saved: {output_path}")

def create_summary_stats(data, output_dir):
    """Create summary statistics text file."""
    years, lang_data, sorted_langs, year_lang_counts = prepare_chart_data(data)
    total_papers = len(data)
    
    output_path = output_dir / 'language_statistics.txt'
    with open(output_path, 'w') as f:
        f.write("Programming Language Usage Statistics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total papers with GitHub repositories: {total_papers}\n")
        f.write(f"Year range: {min(years)} - {max(years)}\n")
        f.write(f"Total unique languages/tools: {len(sorted_langs)}\n\n")
        
        f.write("Language Usage (Total Papers):\n")
        f.write("-" * 50 + "\n")
        for lang in sorted_langs:
            count = sum(lang_data[lang])
            percentage = (count / total_papers) * 100
            f.write(f"{lang:15s}: {count:4d} papers ({percentage:5.1f}%)\n")
        
        f.write("\n\nLanguage Usage by Year:\n")
        f.write("-" * 50 + "\n")
        for year in years:
            f.write(f"\n{year}:\n")
            year_langs = sorted(year_lang_counts[year].items(), 
                              key=lambda x: x[1], reverse=True)
            for lang, count in year_langs:
                f.write(f"  {lang:15s}: {count:3d} papers\n")
    
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    print("Extracting GitHub repository data...")
    data = extract_data()
    
    if not data:
        print("No data found!")
        exit(1)
    
    print(f"\nFound {len(data)} papers with GitHub repositories")
    years = sorted(set(e['year'] for e in data))
    print(f"Year range: {min(years)} - {max(years)}")
    
    # Create output directory
    output_dir = Path('data/plots')
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerating visualizations...")
    create_csv_data(data, output_dir)
    create_html_visualization(data, output_dir)
    create_summary_stats(data, output_dir)
    
    print("\nAll visualizations generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"\nOpen {output_dir / 'language_trends.html'} in your browser to view interactive charts!")
