"""Programming language usage trends over time from GitHub repositories.

Moved from scripts/plot_language_trends.py.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from infrastructure.literature.tool_analysis.common import (
    DEFAULT_EXTRACTED_TEXT_DIR,
    DEFAULT_PLOTS_DIR,
    extract_year_from_citation_key,
    find_code_mentions,
    find_github_urls,
)


def extract_data(extracted_text_dir=DEFAULT_EXTRACTED_TEXT_DIR):
    """Extract GitHub repository data with years and languages."""
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
                year = extract_year_from_citation_key(paper_name)

                if year and code_mentions:
                    results.append({
                        'year': year,
                        'languages': code_mentions,
                        'paper': paper_name,
                        'repo_count': len(cleaned_urls)
                    })
        except Exception as e:
            print(f'Error processing {txt_file}: {e}', file=sys.stderr)

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


def run(output_dir=DEFAULT_PLOTS_DIR, extracted_text_dir=DEFAULT_EXTRACTED_TEXT_DIR):
    """Run the full language-trends visualization pipeline."""
    output_dir = Path(output_dir)

    print("Extracting GitHub repository data...")
    data = extract_data(extracted_text_dir)

    if not data:
        print("No data found!")
        sys.exit(1)

    print(f"\nFound {len(data)} papers with GitHub repositories")
    years = sorted(set(e['year'] for e in data))
    print(f"Year range: {min(years)} - {max(years)}")

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    print("\nGenerating visualizations...")
    create_csv_data(data, output_dir)
    create_html_visualization(data, output_dir)
    create_summary_stats(data, output_dir)

    print("\nAll visualizations generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"\nOpen {output_dir / 'language_trends.html'} in your browser to view interactive charts!")
