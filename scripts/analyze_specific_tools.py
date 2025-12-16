#!/usr/bin/env python3
"""
Analyze specific libraries, frameworks, and tools (beyond languages) used across repositories.
"""

import re
import json
from pathlib import Path
from collections import Counter, defaultdict

# Pattern to find GitHub URLs
github_pattern = re.compile(r'https?://(?:www\.)?github\.com/[^\s\)]+|github\.com/[^\s\)]+', re.IGNORECASE)

# Specific tools, libraries, and frameworks to search for
specific_tools = {
    # Python ML/DL frameworks
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
    # Note: Removed generic "AIF" pattern as it matches theoretical framework mentions
    # Focus on actual software packages instead
    
    # Reinforcement Learning
    'OpenAI Gym': re.compile(r'\bgym\b|\bopenai[_\s]?gym\b', re.IGNORECASE),
    'Stable-Baselines': re.compile(r'\bstable[_\s-]?baselines\b', re.IGNORECASE),
    'Stable-Baselines3': re.compile(r'\bstable[_\s-]?baselines3\b', re.IGNORECASE),
    
    # Computer Vision
    'OpenCV': re.compile(r'\bopencv\b', re.IGNORECASE),
    
    # Robotics
    'ROS': re.compile(r'\bros\b(?!2)', re.IGNORECASE),  # ROS but not ROS2
    'ROS2': re.compile(r'\bros2\b', re.IGNORECASE),
    
    # Development tools
    'Jupyter': re.compile(r'\bjupyter\b', re.IGNORECASE),
    'IPython': re.compile(r'\bipython\b', re.IGNORECASE),
    
    # Web frameworks (less common in research)
    'Django': re.compile(r'\bdjango\b', re.IGNORECASE),
    'Flask': re.compile(r'\bflask\b', re.IGNORECASE),
    
    # Other notable tools
    'Unity': re.compile(r'\bunity\b', re.IGNORECASE),
    'Gazebo': re.compile(r'\bgazebo\b', re.IGNORECASE),
    'MuJoCo': re.compile(r'\bmujoco\b', re.IGNORECASE),
    'PyBullet': re.compile(r'\bpybullet\b', re.IGNORECASE),
    
    # R packages (common ones)
    'ggplot2': re.compile(r'\bggplot2\b', re.IGNORECASE),
    'dplyr': re.compile(r'\bdplyr\b', re.IGNORECASE),
    'tidyverse': re.compile(r'\btidyverse\b', re.IGNORECASE),
    
    # MATLAB toolboxes
    'MATLAB Toolbox': re.compile(r'\bmatlab[_\s]?toolbox\b', re.IGNORECASE),
    'Psychtoolbox': re.compile(r'\bpsychtoolbox\b', re.IGNORECASE),
    
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

def find_tools_in_content(content, github_urls, lines):
    """Find specific tools/libraries mentioned in the content."""
    tools_found = set()
    content_lower = content.lower()
    
    # Check full content
    for tool_name, pattern in specific_tools.items():
        if pattern.search(content_lower):
            tools_found.add(tool_name)
    
    # Also check context around GitHub URLs
    for i, line in enumerate(lines):
        if any(url.replace('https://', '').replace('http://', '') in line.lower() 
               for url in github_urls):
            context_start = max(0, i-15)
            context_end = min(len(lines), i+15)
            context = ' '.join(lines[context_start:context_end]).lower()
            
            for tool_name, pattern in specific_tools.items():
                if pattern.search(context):
                    tools_found.add(tool_name)
    
    return tools_found

def extract_data():
    """Extract tool usage data from extracted text files."""
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
                
                tools_found = find_tools_in_content(content, cleaned_urls, lines)
                paper_name = txt_file.stem
                
                if tools_found:
                    results.append({
                        'tools': tools_found,
                        'paper': paper_name,
                        'repo_count': len(cleaned_urls),
                        'urls': cleaned_urls
                    })
        except Exception as e:
            print(f'Error processing {txt_file}: {e}', file=__import__('sys').stderr)
    
    return results

def analyze_tool_usage(data):
    """Analyze tool usage patterns."""
    # Count individual tool usage
    tool_counts = Counter()
    for entry in data:
        for tool in entry['tools']:
            tool_counts[tool] += 1
    
    # Count tool pairs
    pair_counts = Counter()
    for entry in data:
        tools = sorted(entry['tools'])
        for pair in __import__('itertools').combinations(tools, 2):
            pair_counts[pair] += 1
    
    # Count tool combinations
    combination_counts = Counter()
    for entry in data:
        tools = tuple(sorted(entry['tools']))
        combination_counts[tools] += 1
    
    return tool_counts, pair_counts, combination_counts

def create_analysis_report(data, output_dir):
    """Create comprehensive tool analysis report."""
    tool_counts, pair_counts, combination_counts = analyze_tool_usage(data)
    
    output_path = output_dir / 'specific_tools_analysis.txt'
    with open(output_path, 'w') as f:
        f.write("Specific Tools, Libraries, and Frameworks Analysis\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total papers analyzed: {len(data)}\n")
        f.write(f"Total unique tools found: {len(tool_counts)}\n\n")
        
        # Individual tool usage
        f.write("Tool Usage Frequency:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Tool':<30s} {'Papers':<10s} {'Percentage':<10s}\n")
        f.write("-" * 70 + "\n")
        for tool, count in tool_counts.most_common():
            percentage = (count / len(data)) * 100
            f.write(f"{tool:<30s} {count:<10d} {percentage:>6.1f}%\n")
        
        # Most common pairs
        f.write("\n\nMost Common Tool Pairs (Top 20):\n")
        f.write("-" * 70 + "\n")
        for (tool1, tool2), count in pair_counts.most_common(20):
            pair_name = f"{tool1} + {tool2}"
            percentage = (count / len(data)) * 100
            f.write(f"{pair_name:<50s} {count:4d} papers ({percentage:5.1f}%)\n")
        
        # Most common combinations
        f.write("\n\nMost Common Tool Combinations (Top 15):\n")
        f.write("-" * 70 + "\n")
        for combination, count in combination_counts.most_common(15):
            combo_name = " + ".join(combination)
            percentage = (count / len(data)) * 100
            f.write(f"{combo_name:<60s} {count:4d} papers ({percentage:5.1f}%)\n")
        
        # Papers using each tool
        f.write("\n\nPapers Using Each Tool:\n")
        f.write("-" * 70 + "\n")
        tool_to_papers = defaultdict(list)
        for entry in data:
            for tool in entry['tools']:
                tool_to_papers[tool].append(entry['paper'])
        
        for tool in sorted(tool_counts.keys(), key=lambda x: tool_counts[x], reverse=True):
            f.write(f"\n{tool} ({tool_counts[tool]} papers):\n")
            papers = tool_to_papers[tool]
            # Show first 10 papers
            for paper in papers[:10]:
                f.write(f"  - {paper}\n")
            if len(papers) > 10:
                f.write(f"  ... and {len(papers) - 10} more\n")
    
    print(f"Saved: {output_path}")

def create_csv_files(data, output_dir):
    """Create CSV files with tool usage data."""
    import csv
    
    tool_counts, pair_counts, combination_counts = analyze_tool_usage(data)
    
    # Tool usage by paper
    output_path = output_dir / 'specific_tools_by_paper.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Paper', 'Tools', 'Tool Count', 'Repositories'])
        for entry in sorted(data, key=lambda x: x['paper']):
            tools_str = ' + '.join(sorted(entry['tools']))
            writer.writerow([
                entry['paper'],
                tools_str,
                len(entry['tools']),
                entry['repo_count']
            ])
    print(f"Saved: {output_path}")
    
    # Tool frequency
    output_path = output_dir / 'specific_tools_frequency.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Tool', 'Count', 'Percentage'])
        total = len(data)
        for tool, count in tool_counts.most_common():
            percentage = (count / total) * 100
            writer.writerow([tool, count, f"{percentage:.2f}%"])
    print(f"Saved: {output_path}")

def create_html_visualization(data, output_dir):
    """Create HTML visualization."""
    tool_counts, pair_counts, combination_counts = analyze_tool_usage(data)
    
    # Prepare data
    top_tools = [tool for tool, _ in tool_counts.most_common(15)]
    tool_values = [tool_counts[tool] for tool in top_tools]
    
    # Top pairs
    top_pairs = pair_counts.most_common(15)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Specific Tools and Libraries Analysis</title>
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
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
            color: #333;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .chart-container {{
            margin-top: 20px;
        }}
        canvas {{
            max-height: 400px;
        }}
    </style>
</head>
<body>
    <h1>Specific Tools, Libraries & Frameworks Analysis</h1>
    <p class="subtitle">Beyond Programming Languages - Common Software Tools Used Across Repositories</p>
    
    <div class="section">
        <h2>Summary</h2>
        <p><strong>Total papers analyzed:</strong> {len(data)}</p>
        <p><strong>Total unique tools found:</strong> {len(tool_counts)}</p>
        <p><strong>Papers with specific tools:</strong> {len(data)}</p>
    </div>
    
    <div class="section">
        <h2>Tool Usage Frequency</h2>
        <div class="chart-container">
            <canvas id="toolChart"></canvas>
        </div>
    </div>
    
    <div class="section">
        <h2>Most Common Tool Pairs</h2>
        <table>
            <thead>
                <tr>
                    <th>Tool Pair</th>
                    <th>Co-occurrence</th>
                    <th>% of Papers</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for (tool1, tool2), count in top_pairs:
        percentage = (count / len(data)) * 100
        html_content += f"""
                <tr>
                    <td><strong>{tool1}</strong> + <strong>{tool2}</strong></td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
"""
    
    html_content += f"""
            </tbody>
        </table>
    </div>
    
    <script>
        const toolData = {{
            labels: {json.dumps(top_tools)},
            datasets: [{{
                label: 'Number of Papers',
                data: {json.dumps(tool_values)},
                backgroundColor: [
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                    '#aec7e8', '#ffbb78', '#c5b0d5', '#ff9896', '#c49c94'
                ],
                borderColor: [
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                    '#aec7e8', '#ffbb78', '#c5b0d5', '#ff9896', '#c49c94'
                ],
                borderWidth: 1
            }}]
        }};
        
        new Chart(document.getElementById('toolChart'), {{
            type: 'bar',
            data: toolData,
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    title: {{
                        display: true,
                        text: 'Tool Usage Frequency'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Number of Papers'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Tool / Library / Framework'
                        }},
                        ticks: {{
                            maxRotation: 45,
                            minRotation: 45
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
    
    output_path = output_dir / 'specific_tools.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    print("Analyzing specific tools, libraries, and frameworks...")
    data = extract_data()
    
    if not data:
        print("No data found!")
        exit(1)
    
    print(f"\nFound {len(data)} papers with specific tools/libraries")
    
    # Create output directory
    output_dir = Path('data/plots')
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerating analysis...")
    create_analysis_report(data, output_dir)
    create_csv_files(data, output_dir)
    create_html_visualization(data, output_dir)
    
    print("\nAnalysis complete!")
    print(f"Output directory: {output_dir}")
    print(f"\nOpen {output_dir / 'specific_tools.html'} in your browser to view visualizations!")
