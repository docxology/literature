# Tool Analysis Summary

## What We Found

### Significant Insights About Tool Usage

1. **Domain-Specific Research Focus**
   - **pymdp** (21 papers) - Active Inference library shows strong domain specialization
   - **OpenAI Gym** (29 papers) - Standard RL environment platform
   - Research is highly specialized in Active Inference and Robotics

2. **Tool Stacks Reveal Use Cases**
   - **Robotics**: Gazebo + ROS (5 papers) - simulation before deployment
   - **RL Research**: MuJoCo + OpenAI Gym (3 papers) - high-fidelity physics
   - **Deep Learning**: PyTorch dominance (18 papers vs 4 TensorFlow) - research preference

3. **Research Patterns**
   - Heavy use of simulation tools (Unity, Gazebo, MuJoCo)
   - Integration of deep learning with robotics (PyTorch + ROS)
   - Growing adoption of modern frameworks (JAX, ROS2)

### Key Statistics

- **133 papers** with GitHub repositories
- **21 unique tools** identified
- **6 programming languages** detected (R detection likely has false positives)
- **180 unique GitHub repositories** found

### Tool Categories

**Reinforcement Learning:**
- OpenAI Gym (29 papers)
- MuJoCo (4 papers)
- PyBullet (2 papers)

**Robotics:**
- ROS/ROS2 (11 papers)
- Gazebo (7 papers)
- Unity (9 papers)

**Deep Learning:**
- PyTorch (18 papers)
- TensorFlow (4 papers)
- JAX (4 papers)

**Domain-Specific:**
- pymdp (21 papers) - Active Inference
- Flux.jl (5 papers) - Julia ML

---

## Code-Based Analysis Tools

### Available Scripts

1. **`analyze_code_repositories.py`** - Clone and analyze GitHub repos
   ```bash
   python3 scripts/analyze_code_repositories.py --limit 10 --output data/plots/code_analysis
   ```
   
   **What it does:**
   - Clones GitHub repositories
   - Analyzes dependency files (requirements.txt, setup.py, Project.toml)
   - Detects programming languages
   - Analyzes repository structure (tests, docs, examples)
   - Extracts actual dependencies vs. paper mentions

2. **`analyze_specific_tools.py`** - Tool usage frequency
   ```bash
   python3 scripts/analyze_specific_tools.py
   ```

3. **`analyze_tool_cooccurrence.py`** - Tool pairing patterns
   ```bash
   python3 scripts/analyze_tool_cooccurrence.py
   ```

4. **`analyze_tools_and_languages_over_time.py`** - Temporal trends
   ```bash
   python3 scripts/analyze_tools_and_languages_over_time.py
   ```

### Code Analysis Outputs

The code analysis script generates:

1. **`code_analysis_full.json`** - Complete analysis data
   - Repository structure
   - Dependencies by language
   - Detected languages
   - File counts

2. **`code_analysis_summary.md`** - Human-readable summary
   - Language distribution
   - Top dependencies
   - Tool usage from actual code
   - Repository structure statistics

### What Code Analysis Reveals

**Dependency Analysis:**
- Actual package versions used
- Dependency conflicts
- Tool compatibility patterns

**Implementation Patterns:**
- How tools are actually used in code
- Common integration patterns
- Code organization approaches

**Repository Quality:**
- Test coverage
- Documentation
- Example code availability

---

## Recommendations

### For Understanding Tool Usage

1. **Run code analysis on key repositories**
   ```bash
   # Analyze repositories using pymdp
   python3 scripts/analyze_code_repositories.py --limit 20
   ```

2. **Compare paper claims with code**
   - Check if tools mentioned in papers are actually used
   - Identify version mismatches
   - Find undocumented dependencies

3. **Analyze tool integration patterns**
   - How pymdp integrates with PyTorch
   - ROS + PyTorch usage patterns
   - JAX adoption patterns

### For Research Insights

1. **Tool Evolution**
   - Track tool adoption over time
   - Identify emerging tools (JAX, ROS2)
   - Monitor declining tools (TensorFlow, MATLAB)

2. **Domain Patterns**
   - Robotics: Simulation → Control → Real-world
   - RL: Environment → Algorithm → Evaluation
   - Active Inference: Theory → Implementation → Application

3. **Reproducibility**
   - Check repository completeness
   - Identify missing dependencies
   - Find common setup issues

---

## Files Generated

### Analysis Reports
- `data/plots/tool_usage_insights.md` - Comprehensive insights
- `data/plots/tools_and_languages_summary.md` - Temporal analysis
- `data/plots/specific_tools_analysis.txt` - Tool frequency
- `data/plots/tool_cooccurrence_analysis.txt` - Tool pairs

### Visualizations
- `data/plots/specific_tools.html` - Interactive tool charts
- `data/plots/tool_cooccurrence.html` - Co-occurrence visualization
- `data/plots/tools_and_languages_over_time.html` - Temporal trends

### Data Files
- `data/plots/all_github_repos_comprehensive.json` - All GitHub URLs
- `data/github_repositories.md` - Repository listing
- `data/plots/code_analysis_full.json` - Code analysis results (when run)

---

## Next Steps

1. **Run full code analysis** on all repositories (may take time)
2. **Compare dependencies** across papers using same tools
3. **Identify common architectures** and design patterns
4. **Analyze API usage** to understand tool integration
5. **Track tool versions** to understand compatibility

---

## Notes

- **R language detection**: Likely has false positives (98.5% seems too high)
- **GitHub URL extraction**: Some repos may be private or deleted
- **Text extraction**: May miss some tool mentions in PDFs
- **Code analysis**: Requires git and internet access

