#!/usr/bin/env python3
"""
Comprehensive analysis of tools and languages over time.
Creates a single summary document with trends for both.
"""

import re
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

# Pattern to find GitHub URLs
github_pattern = re.compile(r'https?://(?:www\.)?github\.com/[^\s\)]+|github\.com/[^\s\)]+', re.IGNORECASE)

# Programming languages
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

# Specific tools, libraries, and frameworks
specific_tools = {
    # ML/DL frameworks
    'PyTorch': re.compile(r'\bpytorch\b', re.IGNORECASE),
    'TensorFlow': re.compile(r'\btensorflow\b', re.IGNORECASE),
    'Keras': re.compile(r'\bkeras\b', re.IGNORECASE),
    'JAX': re.compile(r'\bjax\b', re.IGNORECASE),
    
    # Python scientific computing
    'NumPy': re.compile(r'\bnumpy\b', re.IGNORECASE),
    'SciPy': re.compile(r'\bscipy\b', re.IGNORECASE),
    'Pandas': re.compile(r'\bpandas\b', re.IGNORECASE),
    'Matplotlib': re.compile(r'\bmatplotlib\b', re.IGNORECASE),
    'Seaborn': re.compile(r'\bseaborn\b', re.IGNORECASE),
    'scikit-learn': re.compile(r'\bscikit-learn\b|\bsklearn\b', re.IGNORECASE),
    
    # Active Inference specific
    'pymdp': re.compile(r'\bpymdp\b', re.IGNORECASE),
    
    # Reinforcement Learning
    'OpenAI Gym': re.compile(r'\bgym\b|\bopenai[_\s]?gym\b', re.IGNORECASE),
    'Stable-Baselines': re.compile(r'\bstable[_\s-]?baselines\b(?!3)', re.IGNORECASE),
    'Stable-Baselines3': re.compile(r'\bstable[_\s-]?baselines3\b', re.IGNORECASE),
    
    # Computer Vision
    'OpenCV': re.compile(r'\bopencv\b', re.IGNORECASE),
    
    # Robotics
    'ROS': re.compile(r'\bros\b(?!2)', re.IGNORECASE),
    'ROS2': re.compile(r'\bros2\b', re.IGNORECASE),
    
    # Development tools
    'Jupyter': re.compile(r'\bjupyter\b', re.IGNORECASE),
    'IPython': re.compile(r'\bipython\b', re.IGNORECASE),
    
    # Simulation/Game engines
    'Unity': re.compile(r'\bunity\b', re.IGNORECASE),
    'Gazebo': re.compile(r'\bgazebo\b', re.IGNORECASE),
    'MuJoCo': re.compile(r'\bmujoco\b', re.IGNORECASE),
    'PyBullet': re.compile(r'\bpybullet\b', re.IGNORECASE),
    
    # R packages
    'ggplot2': re.compile(r'\bggplot2\b', re.IGNORECASE),
    'dplyr': re.compile(r'\bdplyr\b', re.IGNORECASE),
    'tidyverse': re.compile(r'\btidyverse\b', re.IGNORECASE),
    
    # Julia packages
    'Flux.jl': re.compile(r'\bflux\.jl\b|\bflux\b', re.IGNORECASE),
    'Turing.jl': re.compile(r'\bturing\.jl\b', re.IGNORECASE),
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

def find_items_in_content(content, patterns, github_urls, lines):
    """Find items matching patterns in content."""
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
            context_start = max(0, i-15)
            context_end = min(len(lines), i+15)
            context = ' '.join(lines[context_start:context_end]).lower()
            
            for item_name, pattern in patterns.items():
                if pattern.search(context):
                    items_found.add(item_name)
    
    return items_found

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
    """Extract all data with years, languages, and tools."""
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
                
                languages_found = find_items_in_content(content, languages, cleaned_urls, lines)
                tools_found = find_items_in_content(content, specific_tools, cleaned_urls, lines)
                paper_name = txt_file.stem
                year = extract_year_from_citation_key(paper_name)
                
                if year and (languages_found or tools_found):
                    results.append({
                        'year': year,
                        'languages': languages_found,
                        'tools': tools_found,
                        'paper': paper_name,
                        'repo_count': len(cleaned_urls)
                    })
        except Exception as e:
            print(f'Error processing {txt_file}: {e}', file=__import__('sys').stderr)
    
    return results

def prepare_time_series_data(data):
    """Prepare time series data for languages and tools."""
    years = sorted(set(e['year'] for e in data))
    
    # Language data by year
    lang_by_year = defaultdict(lambda: defaultdict(int))
    for entry in data:
        year = entry['year']
        for lang in entry['languages']:
            lang_by_year[year][lang] += 1
    
    # Tool data by year
    tool_by_year = defaultdict(lambda: defaultdict(int))
    for entry in data:
        year = entry['year']
        for tool in entry['tools']:
            tool_by_year[year][tool] += 1
    
    # Get all languages and tools
    all_languages = set()
    for lang_counts in lang_by_year.values():
        all_languages.update(lang_counts.keys())
    
    all_tools = set()
    for tool_counts in tool_by_year.values():
        all_tools.update(tool_counts.keys())
    
    # Prepare time series
    lang_series = {lang: [lang_by_year[year].get(lang, 0) for year in years] 
                   for lang in all_languages}
    tool_series = {tool: [tool_by_year[year].get(tool, 0) for year in years] 
                   for tool in all_tools}
    
    # Calculate totals for sorting
    lang_totals = {lang: sum(lang_series[lang]) for lang in all_languages}
    tool_totals = {tool: sum(tool_series[tool]) for tool in all_tools}
    
    return years, lang_series, tool_series, lang_totals, tool_totals

def create_comprehensive_summary(data, output_dir):
    """Create comprehensive summary document."""
    years, lang_series, tool_series, lang_totals, tool_totals = prepare_time_series_data(data)
    
    # Sort by total usage
    sorted_langs = sorted(lang_totals.keys(), key=lambda x: lang_totals[x], reverse=True)
    sorted_tools = sorted(tool_totals.keys(), key=lambda x: tool_totals[x], reverse=True)
    
    output_path = output_dir / 'tools_and_languages_summary.md'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Tools and Languages Over Time - Comprehensive Summary\n\n")
        f.write(f"**Analysis Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}\n\n")
        f.write(f"**Total Papers Analyzed:** {len(data)}\n")
        f.write(f"**Year Range:** {min(years)} - {max(years)}\n")
        f.write(f"**Total Unique Languages:** {len(sorted_langs)}\n")
        f.write(f"**Total Unique Tools:** {len(sorted_tools)}\n\n")
        f.write("---\n\n")
        
        # Languages section
        f.write("## Programming Languages Over Time\n\n")
        f.write("### Language Usage Summary\n\n")
        f.write("| Language | Total Papers | Percentage | Trend |\n")
        f.write("|----------|--------------|------------|-------|\n")
        
        for lang in sorted_langs:
            total = lang_totals[lang]
            percentage = (total / len(data)) * 100
            # Calculate trend (comparing last 3 years to first 3 years)
            if len(years) >= 6:
                recent = sum(lang_series[lang][-3:])
                early = sum(lang_series[lang][:3])
                if recent > early * 1.2:
                    trend = "📈 Increasing"
                elif recent < early * 0.8:
                    trend = "📉 Decreasing"
                else:
                    trend = "➡️ Stable"
            else:
                trend = "➡️ N/A"
            f.write(f"| {lang} | {total} | {percentage:.1f}% | {trend} |\n")
        
        f.write("\n### Language Usage by Year\n\n")
        f.write("| Year | Total Papers | " + " | ".join(sorted_langs[:10]) + " |\n")
        f.write("|------|--------------|" + "|".join(["---" for _ in sorted_langs[:10]]) + "|\n")
        for year in years:
            # Count total papers for this year
            year_papers = sum(1 for e in data if e['year'] == year)
            row = [str(year), str(year_papers)]
            for lang in sorted_langs[:10]:
                count = lang_series[lang][years.index(year)]
                row.append(str(count))
            f.write("| " + " | ".join(row) + " |\n")
        
        # Tools section
        f.write("\n---\n\n")
        f.write("## Tools, Libraries & Frameworks Over Time\n\n")
        f.write("### Tool Usage Summary\n\n")
        f.write("| Tool | Total Papers | Percentage | Trend |\n")
        f.write("|------|--------------|------------|-------|\n")
        
        for tool in sorted_tools:
            total = tool_totals[tool]
            percentage = (total / len(data)) * 100
            # Calculate trend
            if len(years) >= 6:
                recent = sum(tool_series[tool][-3:])
                early = sum(tool_series[tool][:3])
                if recent > early * 1.2:
                    trend = "📈 Increasing"
                elif recent < early * 0.8:
                    trend = "📉 Decreasing"
                else:
                    trend = "➡️ Stable"
            else:
                trend = "➡️ N/A"
            f.write(f"| {tool} | {total} | {percentage:.1f}% | {trend} |\n")
        
        f.write("\n### Tool Usage by Year\n\n")
        f.write("| Year | Total Papers | " + " | ".join(sorted_tools[:15]) + " |\n")
        f.write("|------|--------------|" + "|".join(["---" for _ in sorted_tools[:15]]) + "|\n")
        for year in years:
            # Count total papers for this year
            year_papers = sum(1 for e in data if e['year'] == year)
            row = [str(year), str(year_papers)]
            for tool in sorted_tools[:15]:
                count = tool_series[tool][years.index(year)]
                row.append(str(count))
            f.write("| " + " | ".join(row) + " |\n")
        
        # Combined trends
        f.write("\n---\n\n")
        f.write("## Key Insights\n\n")
        
        # Top languages
        f.write("### Most Popular Languages\n")
        for i, lang in enumerate(sorted_langs[:5], 1):
            f.write(f"{i}. **{lang}**: {lang_totals[lang]} papers ({lang_totals[lang]/len(data)*100:.1f}%)\n")
        
        f.write("\n### Most Popular Tools\n")
        for i, tool in enumerate(sorted_tools[:10], 1):
            f.write(f"{i}. **{tool}**: {tool_totals[tool]} papers ({tool_totals[tool]/len(data)*100:.1f}%)\n")
        
        # Growth analysis
        f.write("\n### Growth Trends\n\n")
        f.write("**Fastest Growing Languages (comparing 2023-2025 vs 2016-2018):**\n")
        growth_langs = []
        for lang in sorted_langs:
            if len(years) >= 6:
                recent = sum(lang_series[lang][-3:])
                early = sum(lang_series[lang][:3])
                if early > 0:
                    growth_rate = (recent / early) if early > 0 else 0
                    growth_langs.append((lang, growth_rate, recent, early))
        growth_langs.sort(key=lambda x: x[1], reverse=True)
        for lang, rate, recent, early in growth_langs[:5]:
            if rate > 0:
                f.write(f"- **{lang}**: {rate:.1f}x growth ({early} → {recent} papers)\n")
        
        f.write("\n**Fastest Growing Tools (comparing 2023-2025 vs 2016-2018):**\n")
        growth_tools = []
        for tool in sorted_tools:
            if len(years) >= 6:
                recent = sum(tool_series[tool][-3:])
                early = sum(tool_series[tool][:3])
                if early > 0:
                    growth_rate = (recent / early) if early > 0 else 0
                    growth_tools.append((tool, growth_rate, recent, early))
        growth_tools.sort(key=lambda x: x[1], reverse=True)
        for tool, rate, recent, early in growth_tools[:5]:
            if rate > 0:
                f.write(f"- **{tool}**: {rate:.1f}x growth ({early} → {recent} papers)\n")
    
    print(f"Saved: {output_path}")

def create_csv_data(data, output_dir):
    """Create CSV files with time series data."""
    years, lang_series, tool_series, lang_totals, tool_totals = prepare_time_series_data(data)
    
    sorted_langs = sorted(lang_totals.keys(), key=lambda x: lang_totals[x], reverse=True)
    sorted_tools = sorted(tool_totals.keys(), key=lambda x: tool_totals[x], reverse=True)
    
    # Languages by year
    csv_path = output_dir / 'languages_over_time.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Year'] + sorted_langs)
        for year in years:
            row = [year]
            for lang in sorted_langs:
                row.append(lang_series[lang][years.index(year)])
            writer.writerow(row)
    print(f"Saved: {csv_path}")
    
    # Tools by year
    csv_path = output_dir / 'tools_over_time.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Year'] + sorted_tools)
        for year in years:
            row = [year]
            for tool in sorted_tools:
                row.append(tool_series[tool][years.index(year)])
            writer.writerow(row)
    print(f"Saved: {csv_path}")

def create_html_visualization(data, output_dir):
    """Create HTML visualization with charts."""
    years, lang_series, tool_series, lang_totals, tool_totals = prepare_time_series_data(data)
    
    sorted_langs = sorted(lang_totals.keys(), key=lambda x: lang_totals[x], reverse=True)
    sorted_tools = sorted(tool_totals.keys(), key=lambda x: tool_totals[x], reverse=True)
    
    # Top items for visualization
    top_langs = sorted_langs[:10]
    top_tools = sorted_tools[:12]
    
    years_js = [str(y) for y in years]
    
    # Prepare language datasets
    lang_datasets = []
    colors_lang = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    for i, lang in enumerate(top_langs):
        lang_datasets.append({
            'label': lang,
            'data': lang_series[lang],
            'borderColor': colors_lang[i % len(colors_lang)],
            'backgroundColor': colors_lang[i % len(colors_lang)] + '40',
            'tension': 0.1
        })
    
    # Prepare tool datasets
    tool_datasets = []
    colors_tool = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78'
    ]
    for i, tool in enumerate(top_tools):
        tool_datasets.append({
            'label': tool,
            'data': tool_series[tool],
            'borderColor': colors_tool[i % len(colors_tool)],
            'backgroundColor': colors_tool[i % len(colors_tool)] + '40',
            'tension': 0.1
        })
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tools and Languages Over Time</title>
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
        .section {{
            background: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        .chart-container {{
            margin-top: 20px;
        }}
        canvas {{
            max-height: 500px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-box .number {{
            font-size: 2em;
            font-weight: bold;
            color: #1f77b4;
        }}
        .stat-box .label {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>Tools and Languages Over Time</h1>
    <p class="subtitle">Comprehensive Analysis of Programming Languages and Software Tools ({min(years)} - {max(years)})</p>
    
    <div class="section">
        <div class="stats">
            <div class="stat-box">
                <div class="number">{len(data)}</div>
                <div class="label">Papers Analyzed</div>
            </div>
            <div class="stat-box">
                <div class="number">{len(sorted_langs)}</div>
                <div class="label">Languages</div>
            </div>
            <div class="stat-box">
                <div class="number">{len(sorted_tools)}</div>
                <div class="label">Tools</div>
            </div>
            <div class="stat-box">
                <div class="number">{min(years)}-{max(years)}</div>
                <div class="label">Year Range</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Programming Languages Over Time</h2>
        <div class="chart-container">
            <canvas id="langChart"></canvas>
        </div>
    </div>
    
    <div class="section">
        <h2>Tools, Libraries & Frameworks Over Time</h2>
        <div class="chart-container">
            <canvas id="toolChart"></canvas>
        </div>
    </div>
    
    <script>
        const years = {json.dumps(years_js)};
        const langDatasets = {json.dumps(lang_datasets)};
        const toolDatasets = {json.dumps(tool_datasets)};
        
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
        
        // Language Chart
        new Chart(document.getElementById('langChart'), {{
            type: 'line',
            data: {{
                labels: years,
                datasets: langDatasets
            }},
            options: chartOptions
        }});
        
        // Tool Chart
        new Chart(document.getElementById('toolChart'), {{
            type: 'line',
            data: {{
                labels: years,
                datasets: toolDatasets
            }},
            options: chartOptions
        }});
    </script>
</body>
</html>"""
    
    output_path = output_dir / 'tools_and_languages_over_time.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    print("Analyzing tools and languages over time...")
    data = extract_data()
    
    if not data:
        print("No data found!")
        exit(1)
    
    years = sorted(set(e['year'] for e in data))
    print(f"\nFound {len(data)} papers with GitHub repositories")
    print(f"Year range: {min(years)} - {max(years)}")
    
    # Create output directory
    output_dir = Path('data/plots')
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerating comprehensive analysis...")
    create_comprehensive_summary(data, output_dir)
    create_csv_data(data, output_dir)
    create_html_visualization(data, output_dir)
    
    print("\nAnalysis complete!")
    print(f"Output directory: {output_dir}")
    print(f"\nMain summary document: {output_dir / 'tools_and_languages_summary.md'}")
    print(f"Interactive visualization: {output_dir / 'tools_and_languages_over_time.html'}")
