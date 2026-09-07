"""Tool co-occurrence analysis across GitHub repositories.

Moved from scripts/analyze_tool_cooccurrence.py.
"""

import csv
import json
import sys
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path

from infrastructure.literature.tool_analysis.common import (
    DEFAULT_EXTRACTED_TEXT_DIR,
    DEFAULT_PLOTS_DIR,
    find_code_mentions,
    find_github_urls,
)


def extract_data(extracted_text_dir=DEFAULT_EXTRACTED_TEXT_DIR):
    """Extract GitHub repository data with languages."""
    results = []
    extracted_text_dir = Path(extracted_text_dir)

    if not extracted_text_dir.exists():
        print(f"Error: Directory {extracted_text_dir} does not exist")
        return []

    for txt_file in sorted(extracted_text_dir.glob('*.txt')):
        try:
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')

            cleaned_urls = find_github_urls(content)

            if cleaned_urls:
                code_mentions = find_code_mentions(content, cleaned_urls, lines)
                paper_name = txt_file.stem

                if code_mentions:
                    results.append({
                        'languages': code_mentions,
                        'paper': paper_name,
                        'repo_count': len(cleaned_urls),
                        'urls': cleaned_urls
                    })
        except Exception as e:
            print(f'Error processing {txt_file}: {e}', file=sys.stderr)

    return results


def analyze_cooccurrence(data):
    """Analyze tool co-occurrence patterns."""
    # Count individual tool usage
    tool_counts = Counter()
    for entry in data:
        for lang in entry['languages']:
            tool_counts[lang] += 1

    # Count tool pairs (co-occurrence)
    pair_counts = Counter()
    for entry in data:
        langs = sorted(entry['languages'])  # Sort for consistent pairs
        for pair in combinations(langs, 2):
            pair_counts[pair] += 1

    # Count tool triplets
    triplet_counts = Counter()
    for entry in data:
        langs = sorted(entry['languages'])
        if len(langs) >= 3:
            for triplet in combinations(langs, 3):
                triplet_counts[triplet] += 1

    # Find most common combinations
    return tool_counts, pair_counts, triplet_counts


def calculate_jaccard_similarity(pair_counts, tool_counts):
    """Calculate Jaccard similarity for tool pairs."""
    similarities = {}
    for (tool1, tool2), count in pair_counts.items():
        # Jaccard = intersection / union
        count1 = tool_counts[tool1]
        count2 = tool_counts[tool2]
        union = count1 + count2 - count
        if union > 0:
            jaccard = count / union
            similarities[(tool1, tool2)] = {
                'cooccurrence': count,
                'jaccard': jaccard,
                'tool1_count': count1,
                'tool2_count': count2
            }
    return similarities


def create_cooccurrence_report(data, output_dir):
    """Create comprehensive co-occurrence analysis report."""
    tool_counts, pair_counts, triplet_counts = analyze_cooccurrence(data)
    similarities = calculate_jaccard_similarity(pair_counts, tool_counts)

    output_path = output_dir / 'tool_cooccurrence_analysis.txt'
    with open(output_path, 'w') as f:
        f.write("Tool Co-occurrence Analysis\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total papers analyzed: {len(data)}\n\n")

        # Individual tool usage
        f.write("Individual Tool Usage:\n")
        f.write("-" * 70 + "\n")
        for tool, count in tool_counts.most_common():
            percentage = (count / len(data)) * 100
            f.write(f"{tool:20s}: {count:4d} papers ({percentage:5.1f}%)\n")

        # Most common pairs
        f.write("\n\nMost Common Tool Pairs (Top 20):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Tool Pair':<40s} {'Co-occurrence':<15s} {'Jaccard':<10s}\n")
        f.write("-" * 70 + "\n")
        for (tool1, tool2), info in sorted(similarities.items(),
                                          key=lambda x: x[1]['cooccurrence'],
                                          reverse=True)[:20]:
            pair_name = f"{tool1} + {tool2}"
            f.write(f"{pair_name:<40s} {info['cooccurrence']:<15d} {info['jaccard']:<10.3f}\n")

        # Highest Jaccard similarity (most likely to co-occur)
        f.write("\n\nTool Pairs with Highest Jaccard Similarity (Most Likely to Co-occur):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Tool Pair':<40s} {'Jaccard':<10s} {'Co-occurrence':<15s}\n")
        f.write("-" * 70 + "\n")
        for (tool1, tool2), info in sorted(similarities.items(),
                                          key=lambda x: x[1]['jaccard'],
                                          reverse=True)[:20]:
            pair_name = f"{tool1} + {tool2}"
            f.write(f"{pair_name:<40s} {info['jaccard']:<10.3f} {info['cooccurrence']:<15d}\n")

        # Most common triplets
        f.write("\n\nMost Common Tool Triplets (Top 15):\n")
        f.write("-" * 70 + "\n")
        for triplet, count in triplet_counts.most_common(15):
            triplet_name = " + ".join(triplet)
            percentage = (count / len(data)) * 100
            f.write(f"{triplet_name:<60s} {count:4d} papers ({percentage:5.1f}%)\n")

        # Tool stacks (common combinations)
        f.write("\n\nCommon Tool Stacks:\n")
        f.write("-" * 70 + "\n")

        # Group by number of tools
        stacks_by_size = defaultdict(list)
        for entry in data:
            langs = tuple(sorted(entry['languages']))
            stacks_by_size[len(langs)].append(langs)

        for size in sorted(stacks_by_size.keys()):
            stack_counts = Counter(stacks_by_size[size])
            f.write(f"\n{size}-Tool Stacks (Top 10):\n")
            for stack, count in stack_counts.most_common(10):
                stack_name = " + ".join(stack)
                percentage = (count / len(data)) * 100
                f.write(f"  {stack_name:<60s} {count:4d} papers ({percentage:5.1f}%)\n")

    print(f"Saved: {output_path}")


def create_cooccurrence_matrix_csv(data, output_dir):
    """Create co-occurrence matrix CSV."""
    tool_counts, pair_counts, triplet_counts = analyze_cooccurrence(data)

    # Get all tools sorted by frequency
    all_tools = sorted([tool for tool, _ in tool_counts.most_common()])

    # Create matrix
    matrix = []
    matrix.append([''] + all_tools)  # Header row

    for tool1 in all_tools:
        row = [tool1]
        for tool2 in all_tools:
            if tool1 == tool2:
                # Diagonal: individual count
                row.append(tool_counts[tool1])
            else:
                # Off-diagonal: co-occurrence count
                pair = tuple(sorted([tool1, tool2]))
                row.append(pair_counts.get(pair, 0))
        matrix.append(row)

    # Write CSV
    output_path = output_dir / 'tool_cooccurrence_matrix.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(matrix)
    print(f"Saved: {output_path}")


def create_tool_combinations_csv(data, output_dir):
    """Create CSV with all tool combinations."""
    # Count all unique combinations
    combination_counts = Counter()
    for entry in data:
        langs = tuple(sorted(entry['languages']))
        combination_counts[langs] += 1

    output_path = output_dir / 'tool_combinations.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Tools', 'Count', 'Percentage'])

        total = len(data)
        for combination, count in combination_counts.most_common():
            tools_str = ' + '.join(combination)
            percentage = (count / total) * 100
            writer.writerow([tools_str, count, f"{percentage:.2f}%"])

    print(f"Saved: {output_path}")


def create_html_visualization(data, output_dir):
    """Create HTML visualization of co-occurrence."""
    tool_counts, pair_counts, triplet_counts = analyze_cooccurrence(data)
    similarities = calculate_jaccard_similarity(pair_counts, tool_counts)

    # Get top tools and pairs
    top_tools = [tool for tool, _ in tool_counts.most_common(10)]
    top_pairs = sorted(similarities.items(),
                      key=lambda x: x[1]['cooccurrence'],
                      reverse=True)[:15]

    # Prepare data for visualization
    pair_data = []
    for (tool1, tool2), info in top_pairs:
        pair_data.append({
            'tool1': tool1,
            'tool2': tool2,
            'count': info['cooccurrence'],
            'jaccard': info['jaccard']
        })

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tool Co-occurrence Analysis</title>
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
    <h1>Tool Co-occurrence Analysis</h1>
    <p class="subtitle">Common Tooling Patterns Across GitHub Repositories</p>

    <div class="section">
        <h2>Summary</h2>
        <p><strong>Total papers analyzed:</strong> {len(data)}</p>
        <p><strong>Total unique tools:</strong> {len(tool_counts)}</p>
        <p><strong>Total tool pairs:</strong> {len(pair_counts)}</p>
        <p><strong>Total tool triplets:</strong> {len(triplet_counts)}</p>
    </div>

    <div class="section">
        <h2>Most Common Tool Pairs</h2>
        <table>
            <thead>
                <tr>
                    <th>Tool Pair</th>
                    <th>Co-occurrence Count</th>
                    <th>Jaccard Similarity</th>
                    <th>% of Papers</th>
                </tr>
            </thead>
            <tbody>
"""

    for pair_info in pair_data[:20]:
        percentage = (pair_info['count'] / len(data)) * 100
        html_content += f"""
                <tr>
                    <td><strong>{pair_info['tool1']}</strong> + <strong>{pair_info['tool2']}</strong></td>
                    <td>{pair_info['count']}</td>
                    <td>{pair_info['jaccard']:.3f}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
"""

    html_content += """
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>Individual Tool Usage</h2>
        <div class="chart-container">
            <canvas id="toolChart"></canvas>
        </div>
    </div>

    <script>
        const toolData = {
"""

    # Add tool usage data
    tool_labels = []
    tool_values = []
    for tool, count in tool_counts.most_common(12):
        tool_labels.append(tool)
        tool_values.append(count)

    # Build JavaScript separately to avoid f-string conflicts
    js_data = json.dumps({
        'labels': tool_labels,
        'values': tool_values
    })

    html_content += f"""
            labels: {json.dumps(tool_labels)},
            datasets: [{{
                label: 'Number of Papers',
                data: {json.dumps(tool_values)},
                backgroundColor: [
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                    '#aec7e8', '#ffbb78'
                ],
                borderColor: [
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                    '#aec7e8', '#ffbb78'
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
                            text: 'Programming Language / Tool'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    output_path = output_dir / 'tool_cooccurrence.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Saved: {output_path}")


def run(output_dir=DEFAULT_PLOTS_DIR, extracted_text_dir=DEFAULT_EXTRACTED_TEXT_DIR):
    """Run the full tool co-occurrence analysis pipeline."""
    output_dir = Path(output_dir)

    print("Analyzing tool co-occurrence patterns...")
    data = extract_data(extracted_text_dir)

    if not data:
        print("No data found!")
        sys.exit(1)

    print(f"\nFound {len(data)} papers with GitHub repositories")

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    print("\nGenerating co-occurrence analysis...")
    create_cooccurrence_report(data, output_dir)
    create_cooccurrence_matrix_csv(data, output_dir)
    create_tool_combinations_csv(data, output_dir)
    create_html_visualization(data, output_dir)

    print("\nAnalysis complete!")
    print(f"Output directory: {output_dir}")
    print(f"\nOpen {output_dir / 'tool_cooccurrence.html'} in your browser to view visualizations!")
