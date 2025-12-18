#!/usr/bin/env python3
"""
Code-based analysis of GitHub repositories.
Clones repos, analyzes dependencies, code structure, and usage patterns.
"""

import json
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional
import os

# Dependency file patterns
DEPENDENCY_FILES = {
    'Python': ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile', 'environment.yml', 'conda.yml'],
    'R': ['DESCRIPTION', 'NAMESPACE', '*.Rproj'],
    'Julia': ['Project.toml', 'Manifest.toml'],
    'JavaScript': ['package.json', 'package-lock.json', 'yarn.lock'],
    'Java': ['pom.xml', 'build.gradle', 'build.gradle.kts'],
}

# Common dependency patterns
PYTHON_DEPS = [
    'torch', 'pytorch', 'tensorflow', 'jax', 'numpy', 'scipy', 'pandas',
    'matplotlib', 'seaborn', 'sklearn', 'scikit-learn', 'gym', 'opencv',
    'pymdp', 'stable-baselines', 'stable-baselines3', 'mujoco', 'pybullet'
]

def load_github_repos() -> List[Dict]:
    """Load GitHub repository data from comprehensive JSON."""
    json_file = Path('data/plots/all_github_repos_comprehensive.json')
    if json_file.exists():
        with open(json_file, 'r') as f:
            data = json.load(f)
            return data.get('papers', [])
    
    # Fallback: extract from markdown
    md_file = Path('data/github_repositories.md')
    if md_file.exists():
        return extract_from_markdown(md_file)
    
    return []

def extract_from_markdown(md_file: Path) -> List[Dict]:
    """Extract repo data from markdown file."""
    results = []
    current_paper = None
    
    with open(md_file, 'r') as f:
        content = f.read()
    
    # Simple extraction - could be improved
    paper_blocks = re.split(r'### (\w+)', content)
    for i in range(1, len(paper_blocks), 2):
        if i + 1 < len(paper_blocks):
            paper_key = paper_blocks[i]
            paper_content = paper_blocks[i + 1]
            
            # Extract URLs
            urls = re.findall(r'https?://github\.com/[^\s\)]+', paper_content)
            if urls:
                results.append({
                    'paper': paper_key,
                    'urls': urls
                })
    
    return results

def clone_repo(url: str, temp_dir: Path) -> Optional[Path]:
    """Clone a GitHub repository to temporary directory."""
    try:
        # Extract repo name
        match = re.search(r'github\.com/([^/]+/[^/]+)', url)
        if not match:
            return None
        
        repo_path = temp_dir / match.group(1).replace('/', '_')
        
        # Clone (shallow, single branch)
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', '--single-branch', url, str(repo_path)],
            capture_output=True,
            timeout=60,
            text=True
        )
        
        if result.returncode == 0:
            return repo_path
        else:
            print(f"Failed to clone {url}: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print(f"Timeout cloning {url}")
        return None
    except Exception as e:
        print(f"Error cloning {url}: {e}")
        return None

def analyze_python_dependencies(repo_path: Path) -> Dict:
    """Analyze Python dependencies."""
    deps = {
        'requirements': [],
        'setup_py': [],
        'pyproject': [],
        'imports': set()
    }
    
    # Check requirements.txt
    req_file = repo_path / 'requirements.txt'
    if req_file.exists():
        with open(req_file, 'r') as f:
            for line in f:
                line = line.strip().split('#')[0].strip()
                if line and not line.startswith('-'):
                    # Extract package name
                    pkg = re.split(r'[>=<!=]', line)[0].strip().lower()
                    if pkg:
                        deps['requirements'].append(pkg)
    
    # Check setup.py
    setup_file = repo_path / 'setup.py'
    if setup_file.exists():
        try:
            content = setup_file.read_text()
            # Simple regex for install_requires
            matches = re.findall(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
            for match in matches:
                deps_list = re.findall(r'["\']([^"\']+)["\']', match)
                deps['setup_py'].extend([p.lower() for p in deps_list])
        except:
            pass
    
    # Check pyproject.toml
    pyproject_file = repo_path / 'pyproject.toml'
    if pyproject_file.exists():
        try:
            content = pyproject_file.read_text()
            # Simple extraction
            matches = re.findall(r'["\']([^"\']+)["\']', content)
            deps['pyproject'].extend([m.lower() for m in matches if '/' not in m])
        except:
            pass
    
    # Scan for imports in Python files
    for py_file in repo_path.rglob('*.py'):
        try:
            content = py_file.read_text(encoding='utf-8', errors='ignore')
            # Find import statements
            imports = re.findall(r'^(?:from|import)\s+([a-zA-Z0-9_]+)', content, re.MULTILINE)
            deps['imports'].update([imp.split('.')[0].lower() for imp in imports])
        except:
            pass
    
    return deps

def analyze_r_dependencies(repo_path: Path) -> Dict:
    """Analyze R dependencies."""
    deps = {
        'description': [],
        'imports': set()
    }
    
    # Check DESCRIPTION file
    desc_file = repo_path / 'DESCRIPTION'
    if desc_file.exists():
        try:
            content = desc_file.read_text()
            # Extract Depends/Imports
            matches = re.findall(r'(?:Depends|Imports):\s*(.+?)(?:\n[A-Z]|$)', content, re.DOTALL)
            for match in matches:
                pkgs = re.findall(r'([a-zA-Z0-9.]+)', match)
                deps['description'].extend(pkgs)
        except:
            pass
    
    # Scan R files for library() calls
    for r_file in repo_path.rglob('*.R'):
        try:
            content = r_file.read_text(encoding='utf-8', errors='ignore')
            libs = re.findall(r'library\(["\']?([^"\')]+)["\']?\)', content, re.IGNORECASE)
            deps['imports'].update([lib.lower() for lib in libs])
        except:
            pass
    
    return deps

def analyze_julia_dependencies(repo_path: Path) -> Dict:
    """Analyze Julia dependencies."""
    deps = {
        'project': [],
        'imports': set()
    }
    
    # Check Project.toml
    project_file = repo_path / 'Project.toml'
    if project_file.exists():
        try:
            content = project_file.read_text()
            # Extract dependencies
            matches = re.findall(r'\[deps\]\s*(.*?)(?=\[|$)', content, re.DOTALL)
            for match in matches:
                pkgs = re.findall(r'([a-zA-Z0-9_]+)\s*=', match)
                deps['project'].extend(pkgs)
        except:
            pass
    
    # Scan Julia files
    for jl_file in repo_path.rglob('*.jl'):
        try:
            content = jl_file.read_text(encoding='utf-8', errors='ignore')
            imports = re.findall(r'using\s+([a-zA-Z0-9.]+)', content)
            deps['imports'].update([imp.split('.')[0].lower() for imp in imports])
        except:
            pass
    
    return deps

def detect_language(repo_path: Path) -> List[str]:
    """Detect programming languages in repository."""
    languages = set()
    
    # Count files by extension
    extensions = Counter()
    for file in repo_path.rglob('*'):
        if file.is_file():
            ext = file.suffix.lower()
            extensions[ext] += 1
    
    # Language detection based on file extensions
    if extensions.get('.py', 0) > 0:
        languages.add('Python')
    if extensions.get('.r', 0) > 0 or extensions.get('.R', 0) > 0:
        languages.add('R')
    if extensions.get('.jl', 0) > 0:
        languages.add('Julia')
    if extensions.get('.js', 0) > 0 or extensions.get('.ts', 0) > 0:
        languages.add('JavaScript')
    if extensions.get('.java', 0) > 0:
        languages.add('Java')
    if extensions.get('.cpp', 0) > 0 or extensions.get('.cc', 0) > 0 or extensions.get('.cxx', 0) > 0:
        languages.add('C++')
    if extensions.get('.m', 0) > 0 and (repo_path / 'README.md').exists():
        # Could be MATLAB or Objective-C, check context
        languages.add('MATLAB')
    
    return list(languages)

def analyze_data_files(repo_path: Path) -> Dict:
    """Analyze data files and data-related code patterns."""
    data_analysis = {
        'data_files': {
            'csv': 0, 'json': 0, 'npy': 0, 'npz': 0, 'h5': 0, 'hdf5': 0,
            'pkl': 0, 'pickle': 0, 'mat': 0, 'txt': 0, 'tsv': 0,
            'parquet': 0, 'feather': 0, 'arrow': 0
        },
        'data_dirs': [],
        'dataset_references': [],
        'data_loading_patterns': [],
        'evaluation_metrics': set(),
        'config_files': [],
        'results_files': [],
        'seed_usage': False,
        'visualization_code': False
    }
    
    # Find data files
    data_extensions = {
        '.csv': 'csv', '.json': 'json', '.npy': 'npy', '.npz': 'npz',
        '.h5': 'h5', '.hdf5': 'hdf5', '.pkl': 'pkl', '.pickle': 'pickle',
        '.mat': 'mat', '.txt': 'txt', '.tsv': 'tsv', '.parquet': 'parquet',
        '.feather': 'feather', '.arrow': 'arrow'
    }
    
    for file in repo_path.rglob('*'):
        if file.is_file():
            ext = file.suffix.lower()
            if ext in data_extensions:
                data_analysis['data_files'][data_extensions[ext]] += 1
    
    # Find data directories
    data_dir_names = ['data', 'dataset', 'datasets', 'input', 'inputs', 'raw_data', 
                      'processed_data', 'results', 'outputs', 'experiments']
    for dir_name in data_dir_names:
        if (repo_path / dir_name).exists():
            data_analysis['data_dirs'].append(dir_name)
    
    # Scan code files for data patterns
    code_files = list(repo_path.rglob('*.py')) + list(repo_path.rglob('*.R')) + \
                 list(repo_path.rglob('*.jl')) + list(repo_path.rglob('*.m'))
    
    for code_file in code_files:
        try:
            content = code_file.read_text(encoding='utf-8', errors='ignore')
            content_lower = content.lower()
            
            # Data loading patterns
            loading_patterns = [
                r'pd\.read_csv|pandas\.read_csv|read\.csv',
                r'np\.load|numpy\.load|load\(.*\.npy',
                r'pickle\.load|joblib\.load',
                r'h5py\.File|hdf5',
                r'torch\.load|torchvision\.datasets',
                r'tf\.data|tensorflow\.data',
                r'datasets\.load_dataset',  # HuggingFace datasets
                r'sklearn\.datasets',
                r'gym\.make',  # OpenAI Gym environments
            ]
            
            for pattern in loading_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    data_analysis['data_loading_patterns'].append(pattern)
            
            # Dataset references (URLs, dataset names)
            dataset_urls = re.findall(r'https?://[^\s\)]+(?:dataset|data|zenodo|kaggle|drive\.google)[^\s\)]*', content, re.IGNORECASE)
            dataset_names = re.findall(r'(?:dataset|data)\s*[=:]\s*["\']([^"\']+)["\']', content, re.IGNORECASE)
            data_analysis['dataset_references'].extend(dataset_urls)
            data_analysis['dataset_references'].extend(dataset_names)
            
            # Evaluation metrics
            metric_patterns = [
                r'\baccuracy\b', r'\bprecision\b', r'\brecall\b', r'\bf1[_\s]?score\b',
                r'\bloss\b', r'\berror\b', r'\brmse\b', r'\bmae\b', r'\br2[_\s]?score\b',
                r'\bauc\b', r'\broc[_\s]?auc\b', r'\bconfusion[_\s]?matrix\b',
                r'\bperplexity\b', r'\bbleu\b', r'\brouge\b', r'\bmeteor\b',
                r'\bfree[_\s]?energy\b', r'\bvariational[_\s]?free[_\s]?energy\b',
                r'\belbo\b', r'\bkl[_\s]?divergence\b', r'\bentropy\b'
            ]
            
            for pattern in metric_patterns:
                if re.search(pattern, content_lower):
                    metric_name = re.sub(r'[_\s]+', '_', pattern.replace(r'\b', '').replace('\\', ''))
                    data_analysis['evaluation_metrics'].add(metric_name)
            
            # Config files
            if re.search(r'config|yaml|yml|json|toml|ini', code_file.name, re.IGNORECASE):
                if code_file.suffix.lower() in ['.yaml', '.yml', '.json', '.toml', '.ini']:
                    data_analysis['config_files'].append(str(code_file.relative_to(repo_path)))
            
            # Results files
            if re.search(r'result|metric|score|output|eval', code_file.name, re.IGNORECASE):
                if code_file.suffix.lower() in ['.json', '.csv', '.txt']:
                    data_analysis['results_files'].append(str(code_file.relative_to(repo_path)))
            
            # Seed usage (reproducibility)
            if re.search(r'random\.seed|np\.random\.seed|torch\.manual_seed|tf\.random\.set_seed|set\.seed', content, re.IGNORECASE):
                data_analysis['seed_usage'] = True
            
            # Visualization code
            if re.search(r'matplotlib|plt\.|seaborn|plotly|bokeh|visualize|plot\(', content, re.IGNORECASE):
                data_analysis['visualization_code'] = True
                
        except:
            pass
    
    # Remove duplicates
    data_analysis['dataset_references'] = list(set(data_analysis['dataset_references']))
    data_analysis['data_loading_patterns'] = list(set(data_analysis['data_loading_patterns']))
    
    return data_analysis

def analyze_experimental_setup(repo_path: Path) -> Dict:
    """Analyze experimental setup and reproducibility elements."""
    setup = {
        'has_config': False,
        'config_format': None,
        'hyperparameters': set(),
        'experiment_tracking': False,
        'wandb': False,
        'tensorboard': False,
        'mlflow': False,
        'comet': False,
        'reproducibility': {
            'seed_setting': False,
            'environment_file': False,
            'docker': False,
            'conda': False
        }
    }
    
    # Check for config files
    config_files = list(repo_path.rglob('config*.yaml')) + \
                   list(repo_path.rglob('config*.yml')) + \
                   list(repo_path.rglob('config*.json')) + \
                   list(repo_path.rglob('config*.toml'))
    
    if config_files:
        setup['has_config'] = True
        setup['config_format'] = config_files[0].suffix.lower()
    
    # Check for environment files
    env_files = ['environment.yml', 'conda.yml', 'requirements.txt', 'Pipfile', 
                 'Dockerfile', 'docker-compose.yml']
    for env_file in env_files:
        if (repo_path / env_file).exists():
            setup['reproducibility']['environment_file'] = True
            if 'docker' in env_file.lower():
                setup['reproducibility']['docker'] = True
            if 'conda' in env_file.lower():
                setup['reproducibility']['conda'] = True
    
    # Scan code for experiment tracking
    code_files = list(repo_path.rglob('*.py'))
    for code_file in code_files:
        try:
            content = code_file.read_text(encoding='utf-8', errors='ignore')
            
            # Experiment tracking tools
            if 'wandb' in content.lower() or 'import wandb' in content:
                setup['wandb'] = True
                setup['experiment_tracking'] = True
            if 'tensorboard' in content.lower() or 'SummaryWriter' in content:
                setup['tensorboard'] = True
                setup['experiment_tracking'] = True
            if 'mlflow' in content.lower():
                setup['mlflow'] = True
                setup['experiment_tracking'] = True
            if 'comet' in content.lower():
                setup['comet'] = True
                setup['experiment_tracking'] = True
            
            # Hyperparameters
            hp_patterns = [
                r'learning_rate\s*[=:]\s*([0-9.e-]+)',
                r'batch_size\s*[=:]\s*(\d+)',
                r'epochs?\s*[=:]\s*(\d+)',
                r'hidden[_\s]?size\s*[=:]\s*(\d+)',
                r'lr\s*[=:]\s*([0-9.e-]+)'
            ]
            
            for pattern in hp_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    setup['hyperparameters'].add(pattern.split('\\')[0])
            
            # Seed setting
            if re.search(r'random\.seed|np\.random\.seed|torch\.manual_seed', content, re.IGNORECASE):
                setup['reproducibility']['seed_setting'] = True
                
        except:
            pass
    
    return setup

def analyze_repo_structure(repo_path: Path) -> Dict:
    """Analyze repository structure and organization."""
    structure = {
        'has_readme': False,
        'has_license': False,
        'has_tests': False,
        'has_docs': False,
        'has_examples': False,
        'file_count': 0,
        'code_files': 0,
        'languages': []
    }
    
    # Check for common files
    if (repo_path / 'README.md').exists() or (repo_path / 'README.rst').exists():
        structure['has_readme'] = True
    
    for license_file in ['LICENSE', 'LICENSE.txt', 'LICENSE.md']:
        if (repo_path / license_file).exists():
            structure['has_license'] = True
            break
    
    # Check for test directories
    test_dirs = ['tests', 'test', 'spec', '__tests__']
    for test_dir in test_dirs:
        if (repo_path / test_dir).exists():
            structure['has_tests'] = True
            break
    
    # Check for docs
    doc_dirs = ['docs', 'doc', 'documentation']
    for doc_dir in doc_dirs:
        if (repo_path / doc_dir).exists():
            structure['has_docs'] = True
            break
    
    # Check for examples
    example_dirs = ['examples', 'example', 'demos', 'demo']
    for example_dir in example_dirs:
        if (repo_path / example_dir).exists():
            structure['has_examples'] = True
            break
    
    # Count files
    code_extensions = {'.py', '.r', '.R', '.jl', '.js', '.ts', '.java', '.cpp', '.cc', '.cxx', '.m'}
    for file in repo_path.rglob('*'):
        if file.is_file():
            structure['file_count'] += 1
            if file.suffix.lower() in code_extensions:
                structure['code_files'] += 1
    
    structure['languages'] = detect_language(repo_path)
    
    return structure

def analyze_repository(url: str, temp_dir: Path) -> Optional[Dict]:
    """Analyze a single repository."""
    repo_path = clone_repo(url, temp_dir)
    if not repo_path:
        return None
    
    try:
        analysis = {
            'url': url,
            'structure': analyze_repo_structure(repo_path),
            'dependencies': {},
            'data_analysis': analyze_data_files(repo_path),
            'experimental_setup': analyze_experimental_setup(repo_path)
        }
        
        # Analyze dependencies by language
        languages = analysis['structure']['languages']
        
        if 'Python' in languages:
            analysis['dependencies']['python'] = analyze_python_dependencies(repo_path)
        
        if 'R' in languages:
            analysis['dependencies']['r'] = analyze_r_dependencies(repo_path)
        
        if 'Julia' in languages:
            analysis['dependencies']['julia'] = analyze_julia_dependencies(repo_path)
        
        return analysis
    finally:
        # Clean up
        if repo_path.exists():
            shutil.rmtree(repo_path, ignore_errors=True)

def create_dependency_summary(analyses: List[Dict]) -> Dict:
    """Create summary of dependencies across all repositories."""
    all_deps = {
        'python': Counter(),
        'r': Counter(),
        'julia': Counter(),
        'languages': Counter(),
        'tools': Counter()
    }
    
    # Track specific tools
    tool_patterns = {
        'pytorch': ['torch', 'pytorch'],
        'tensorflow': ['tensorflow'],
        'jax': ['jax'],
        'numpy': ['numpy'],
        'scipy': ['scipy'],
        'gym': ['gym'],
        'pymdp': ['pymdp'],
        'mujoco': ['mujoco'],
        'ros': ['ros'],
        'opencv': ['opencv', 'cv2']
    }
    
    for analysis in analyses:
        if not analysis:
            continue
        
        # Count languages
        for lang in analysis.get('structure', {}).get('languages', []):
            all_deps['languages'][lang] += 1
        
        # Count Python dependencies
        py_deps = analysis.get('dependencies', {}).get('python', {})
        for dep_list in [py_deps.get('requirements', []), py_deps.get('setup_py', []), 
                         py_deps.get('pyproject', []), list(py_deps.get('imports', []))]:
            for dep in dep_list:
                all_deps['python'][dep] += 1
                
                # Check for specific tools
                for tool, patterns in tool_patterns.items():
                    if any(p in dep for p in patterns):
                        all_deps['tools'][tool] += 1
        
        # Count R dependencies
        r_deps = analysis.get('dependencies', {}).get('r', {})
        for dep in r_deps.get('description', []):
            all_deps['r'][dep] += 1
        for dep in r_deps.get('imports', []):
            all_deps['r'][dep] += 1
    
    return all_deps

def generate_report(analyses: List[Dict], summary: Dict, output_dir: Path):
    """Generate comprehensive analysis report."""
    output_dir.mkdir(exist_ok=True)
    
    # Convert sets to lists for JSON serialization
    def convert_sets(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_sets(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_sets(item) for item in obj]
        return obj
    
    analyses_serializable = convert_sets(analyses)
    
    # Save full analysis
    with open(output_dir / 'code_analysis_full.json', 'w') as f:
        json.dump(analyses_serializable, f, indent=2)
    
    # Generate summary report
    report_path = output_dir / 'code_analysis_summary.md'
    valid_analyses = [a for a in analyses if a]
    
    with open(report_path, 'w') as f:
        f.write("# Code-Based Repository Analysis\n\n")
        f.write(f"**Total repositories analyzed:** {len(valid_analyses)}\n\n")
        
        f.write("## Language Distribution\n\n")
        f.write("| Language | Repositories |\n")
        f.write("|----------|-------------|\n")
        for lang, count in summary['languages'].most_common():
            f.write(f"| {lang} | {count} |\n")
        
        f.write("\n## Top Python Dependencies\n\n")
        f.write("| Package | Frequency |\n")
        f.write("|---------|----------|\n")
        for pkg, count in summary['python'].most_common(20):
            f.write(f"| {pkg} | {count} |\n")
        
        f.write("\n## Tool Usage (from dependencies)\n\n")
        f.write("| Tool | Frequency |\n")
        f.write("|------|----------|\n")
        for tool, count in summary['tools'].most_common():
            f.write(f"| {tool} | {count} |\n")
        
        f.write("\n## Repository Structure Analysis\n\n")
        structures = [a.get('structure', {}) for a in valid_analyses]
        if structures:
            f.write(f"- Repositories with README: {sum(1 for s in structures if s.get('has_readme'))}\n")
            f.write(f"- Repositories with LICENSE: {sum(1 for s in structures if s.get('has_license'))}\n")
            f.write(f"- Repositories with tests: {sum(1 for s in structures if s.get('has_tests'))}\n")
            f.write(f"- Repositories with docs: {sum(1 for s in structures if s.get('has_docs'))}\n")
            f.write(f"- Repositories with examples: {sum(1 for s in structures if s.get('has_examples'))}\n")
        
        # Empirical Analysis Section
        f.write("\n---\n\n")
        f.write("## Empirical Analysis & Reproducibility\n\n")
        
        data_analyses = [a.get('data_analysis', {}) for a in valid_analyses]
        setups = [a.get('experimental_setup', {}) for a in valid_analyses]
        
        # Data files
        f.write("### Data Files Found\n\n")
        total_data_files = Counter()
        for da in data_analyses:
            for file_type, count in da.get('data_files', {}).items():
                if count > 0:
                    total_data_files[file_type] += count
        
        if total_data_files:
            f.write("| File Type | Total Files |\n")
            f.write("|-----------|-------------|\n")
            for file_type, count in total_data_files.most_common():
                f.write(f"| {file_type.upper()} | {count} |\n")
        else:
            f.write("No data files detected.\n")
        
        # Data loading patterns
        f.write("\n### Data Loading Patterns\n\n")
        loading_patterns = Counter()
        for da in data_analyses:
            for pattern in da.get('data_loading_patterns', []):
                loading_patterns[pattern] += 1
        
        if loading_patterns:
            f.write("| Pattern | Frequency |\n")
            f.write("|---------|----------|\n")
            for pattern, count in loading_patterns.most_common():
                pattern_name = pattern.replace('r\'', '').replace('\\', '').replace('|', ' or ')
                f.write(f"| {pattern_name[:50]} | {count} |\n")
        
        # Evaluation metrics
        f.write("\n### Evaluation Metrics Used\n\n")
        all_metrics = set()
        for da in data_analyses:
            all_metrics.update(da.get('evaluation_metrics', set()))
        
        if all_metrics:
            f.write("| Metric | Repositories |\n")
            f.write("|--------|-------------|\n")
            metric_counts = Counter()
            for da in data_analyses:
                for metric in da.get('evaluation_metrics', set()):
                    metric_counts[metric] += 1
            
            for metric, count in metric_counts.most_common():
                f.write(f"| {metric} | {count} |\n")
        else:
            f.write("No evaluation metrics detected.\n")
        
        # Dataset references
        f.write("\n### Dataset References\n\n")
        all_datasets = []
        for da in data_analyses:
            all_datasets.extend(da.get('dataset_references', []))
        
        if all_datasets:
            unique_datasets = list(set(all_datasets))[:20]  # Top 20
            f.write(f"Found {len(set(all_datasets))} unique dataset references:\n\n")
            for dataset in unique_datasets:
                f.write(f"- {dataset}\n")
        else:
            f.write("No dataset references detected.\n")
        
        # Reproducibility
        f.write("\n### Reproducibility Elements\n\n")
        seed_usage = sum(1 for s in setups if s.get('reproducibility', {}).get('seed_setting'))
        env_files = sum(1 for s in setups if s.get('reproducibility', {}).get('environment_file'))
        docker = sum(1 for s in setups if s.get('reproducibility', {}).get('docker'))
        conda = sum(1 for s in setups if s.get('reproducibility', {}).get('conda'))
        
        f.write(f"- Repositories with seed setting: {seed_usage}/{len(setups)}\n")
        f.write(f"- Repositories with environment files: {env_files}/{len(setups)}\n")
        f.write(f"- Repositories with Docker: {docker}/{len(setups)}\n")
        f.write(f"- Repositories with Conda: {conda}/{len(setups)}\n")
        
        # Experiment tracking
        f.write("\n### Experiment Tracking Tools\n\n")
        wandb_count = sum(1 for s in setups if s.get('wandb'))
        tb_count = sum(1 for s in setups if s.get('tensorboard'))
        mlflow_count = sum(1 for s in setups if s.get('mlflow'))
        
        if wandb_count or tb_count or mlflow_count:
            f.write(f"- Weights & Biases (wandb): {wandb_count}\n")
            f.write(f"- TensorBoard: {tb_count}\n")
            f.write(f"- MLflow: {mlflow_count}\n")
        else:
            f.write("No experiment tracking tools detected.\n")
        
        # Visualization
        viz_count = sum(1 for da in data_analyses if da.get('visualization_code'))
        f.write(f"\n### Visualization Code\n\n")
        f.write(f"Repositories with visualization code: {viz_count}/{len(data_analyses)}\n")
    
    print(f"Saved report to: {report_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze GitHub repositories')
    parser.add_argument('--limit', type=int, default=10, help='Limit number of repos to analyze')
    parser.add_argument('--output', type=str, default='data/plots/code_analysis', help='Output directory')
    args = parser.parse_args()
    
    print("Loading GitHub repository data...")
    repos_data = load_github_repos()
    
    if not repos_data:
        print("No repository data found!")
        exit(1)
    
    print(f"Found {len(repos_data)} papers with GitHub repositories")
    
    # Collect all unique URLs
    all_urls = set()
    for paper in repos_data:
        for url in paper.get('urls', []):
            all_urls.add(url)
    
    print(f"Total unique repositories: {len(all_urls)}")
    print(f"Analyzing first {min(args.limit, len(all_urls))} repositories...")
    
    # Create temporary directory for cloning
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        analyses = []
        for i, url in enumerate(list(all_urls)[:args.limit]):
            print(f"\n[{i+1}/{min(args.limit, len(all_urls))}] Analyzing {url}...")
            analysis = analyze_repository(url, temp_path)
            if analysis:
                analyses.append(analysis)
                print(f"  ✓ Analyzed: {len(analysis.get('structure', {}).get('languages', []))} languages detected")
            else:
                print(f"  ✗ Failed to analyze")
        
        # Generate summary
        print("\nGenerating summary...")
        summary = create_dependency_summary(analyses)
        
        # Save reports
        output_dir = Path(args.output)
        generate_report(analyses, summary, output_dir)
        
        print(f"\nAnalysis complete!")
        print(f"Successfully analyzed: {len(analyses)} repositories")
        print(f"Output directory: {output_dir}")

