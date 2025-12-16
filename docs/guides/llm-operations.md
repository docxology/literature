# LLM Operations Guide

This guide describes the multi-paper LLM operations available for literature analysis.

## Overview

All LLM operations use local Ollama models to generate text-based analyses. These operations synthesize information across multiple papers in your library to produce structured outputs.

## Available Operations

### 1. Summarize Papers

**Purpose:** Generate comprehensive summaries for individual papers with PDFs.

**Input Requirements:**
- Papers must have PDFs downloaded
- Extracted text files are used (from `data/extracted_text/`)
- Prefers existing summaries if available

**Output:**
- **Format:** Markdown files
- **Location:** `data/summaries/{citation_key}_summary.md`
- **Length:** 600-1000 words per paper
- **Content:** Technical summaries with sections for overview, key contributions, methodology, results, and limitations

**Use Cases:**
- Generate summaries for all papers in your library
- Create detailed summaries for papers you haven't read yet
- Build a searchable knowledge base of paper summaries

**Example:**
```bash
python3 scripts/literature_search.py --summarize
```

---

### 2. Literature Review Synthesis

**Purpose:** Synthesize multiple papers into a cohesive literature review paragraph.

**Input Requirements:**
- Multiple papers (default: up to 10)
- Prefers paper summaries, falls back to abstracts
- Configurable focus area (methodology, results, theory, general)

**Output:**
- **Format:** Markdown file
- **Location:** `literature/llm_outputs/review_outputs/literature_review_{timestamp}.md`
- **Length:** 300-500 words
- **Content:** Flowing narrative identifying common themes, comparing approaches, highlighting findings, and noting gaps

**Use Cases:**
- Write literature review sections for manuscripts
- Understand how multiple papers relate to each other
- Identify common themes across a research area

**Example:**
```bash
python3 scripts/literature_search.py --llm-operation review
```

**Configuration:**
- Create `literature/paper_selection.yaml` to select specific papers
- Default focus: "general" (can be "methodology", "results", "theory")

---

### 3. Science Communication Narrative

**Purpose:** Create accessible science communication narratives for general audiences.

**Input Requirements:**
- Multiple papers
- Prefers summaries with key findings sections
- Falls back to abstracts if summaries unavailable

**Output:**
- **Format:** Markdown file
- **Location:** `literature/llm_outputs/communication_outputs/science_communication_{timestamp}.md`
- **Length:** 600-800 words
- **Content:** Engaging narrative explaining scientific concepts in accessible language, connecting research to real-world implications

**Use Cases:**
- Write blog posts or articles about research
- Create educational content
- Explain complex research to non-experts
- Prepare presentations for general audiences

**Example:**
```bash
python3 scripts/literature_search.py --llm-operation communication
```

**Configuration:**
- Audience: "general_public" (default), "students", or "researchers"
- Narrative style: "storytelling" (default), "explanation", or "timeline"

---

### 4. Comparative Analysis

**Purpose:** Compare methods, findings, datasets, or performance across multiple papers.

**Input Requirements:**
- Multiple papers (typically 3-10 for meaningful comparison)
- Prefers full summaries, falls back to abstracts
- Configurable aspect to compare

**Output:**
- **Format:** Markdown file
- **Location:** `literature/llm_outputs/compare_outputs/comparative_analysis_{timestamp}.md`
- **Length:** 500-700 words
- **Content:** Structured analysis with sections for Introduction, Comparison, Analysis, and Conclusions, identifying similarities, differences, strengths, and weaknesses

**Use Cases:**
- Compare different methodological approaches
- Evaluate performance across papers
- Understand trade-offs between different methods
- Identify best practices in a research area

**Example:**
```bash
python3 scripts/literature_search.py --llm-operation compare
```

**Configuration:**
- Aspect: "methods" (default), "results", "datasets", or "performance"

---

### 5. Research Gap Identification

**Purpose:** Identify unanswered questions, methodological gaps, and future research directions.

**Input Requirements:**
- Multiple papers (typically 5-20 for gap analysis)
- Prefers full summaries, falls back to abstracts
- Domain context for focused analysis

**Output:**
- **Format:** Markdown file
- **Location:** `literature/llm_outputs/gaps_outputs/research_gaps_{timestamp}.md`
- **Length:** 400-600 words
- **Content:** Structured analysis with sections for Current State, Identified Gaps, and Recommendations, prioritizing important and feasible research directions

**Use Cases:**
- Identify research opportunities for new projects
- Find methodological limitations in existing work
- Discover inconsistencies or contradictions in literature
- Plan future research directions

**Example:**
```bash
python3 scripts/literature_search.py --llm-operation gaps
```

**Configuration:**
- Domain: "general" (default) or specific domain name for context

---

### 6. Citation Network Analysis

**Purpose:** Analyze intellectual connections and relationships between papers.

**Input Requirements:**
- Multiple papers
- Prefers full summaries, falls back to abstracts

**Output:**
- **Format:** Markdown file
- **Location:** `literature/llm_outputs/network_outputs/citation_network_{timestamp}.md`
- **Length:** 500-700 words
- **Content:** Text-based analysis identifying how papers build upon each other, common methodologies/frameworks, research trajectories, key foundational papers, and implications for the field

**Use Cases:**
- Understand how research ideas evolved
- Identify foundational papers in a field
- Trace research trajectories
- Map intellectual connections between papers

**Example:**
```bash
python3 scripts/literature_search.py --llm-operation network
```

**Important Note:** This operation generates **text-based analysis** of intellectual connections. It does NOT create network graph visualizations. For graph-based citation network visualization, see the meta-analysis module (currently a placeholder feature).

---

## Operation Workflow

### Step 1: Select Papers

All multi-paper operations (2-6) use paper selection configuration:

1. Create `literature/paper_selection.yaml` (optional)
2. If not provided, all papers in library are used
3. Selection criteria include:
   - Citation keys (specific papers)
   - Year range
   - Source filtering
   - PDF/summary availability
   - Keyword matching
   - Limit (maximum papers)

### Step 2: Run Operation

Operations can be run:
- Via interactive menu: `./run_literature.sh` → Option 5.1
- Via command line: `python3 scripts/literature_search.py --llm-operation {operation}`

### Step 3: Review Output

Outputs are saved as markdown files with:
- Operation metadata (papers used, generation time, tokens)
- Operation-specific parameters
- Generated content

---

## Input Quality Considerations

**Best Results When:**
- Papers have generated summaries (operations 2-6 work better with summaries than abstracts)
- Multiple papers are selected (3-10 papers typically optimal)
- Papers are related (same domain, similar topics)

**Fallback Behavior:**
- If summaries unavailable, operations use abstracts
- If abstracts unavailable, operations use titles only (limited quality)

**Recommendation:** Generate summaries first (Operation 1) before running multi-paper operations for best results.

---

## Output Locations Summary

| Operation | Output Directory |
|-----------|------------------|
| Summarize papers | `data/summaries/` |
| Literature review | `literature/llm_outputs/review_outputs/` |
| Science communication | `literature/llm_outputs/communication_outputs/` |
| Comparative analysis | `literature/llm_outputs/compare_outputs/` |
| Research gaps | `literature/llm_outputs/gaps_outputs/` |
| Citation network | `literature/llm_outputs/network_outputs/` |

---

## Citation Network: Text Analysis vs. Visualization

**Important Distinction:**

1. **LLM Citation Network Analysis** (Operation 6) - **Fully Functional**
   - Text-based analysis of intellectual connections
   - Uses LLM to identify relationships, trajectories, and key papers
   - Output: Markdown text file
   - Location: `literature/llm_outputs/network_outputs/`

2. **Citation Network Visualization** (Meta-Analysis Module) - **Placeholder**
   - Would create network graphs showing citation relationships
   - Currently shows "Feature coming soon" message
   - Location: `infrastructure/literature/meta_analysis/additional_visualizations.py`
   - Future feature for graph-based visualization

The LLM operation (6) is a real, functional text analysis tool. The visualization feature is separate and not yet implemented.

---

## Configuration

### Environment Variables

All operations use standard LLM configuration:
- `OLLAMA_HOST` - Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL` - Model to use (default: gemma3:4b)
- `LLM_TEMPERATURE` - Generation temperature (default: 0.7)
- `LLM_MAX_TOKENS` - Maximum tokens per response
- `LLM_TIMEOUT` - Request timeout in seconds

### Paper Selection Config

Create `literature/paper_selection.yaml`:

```yaml
citation_keys: []  # Specific papers by key
year_min: null     # Minimum year
year_max: null     # Maximum year
sources: []        # Filter by source
has_pdf: true      # Require PDF
has_summary: true  # Require summary
keywords: []       # Keywords in title/abstract
limit: 10          # Maximum papers
```

---

## Troubleshooting

**Operation fails with "No papers match selection criteria":**
- Check your `paper_selection.yaml` configuration
- Verify papers exist in library
- Ensure selection criteria aren't too restrictive

**Low quality outputs:**
- Ensure papers have summaries (run Operation 1 first)
- Select related papers (same domain/topic)
- Use appropriate number of papers (3-10 typically optimal)

**Timeout errors:**
- Increase `LLM_TIMEOUT` environment variable
- Use smaller number of papers
- Check Ollama server is running and responsive

---

## See Also

- [Summarize Papers Guide](summarize-papers.md) - Detailed guide for paper summarization
- [Configuration Guide](configuration.md) - LLM and system configuration
- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions
- [Meta-Analysis Guide](meta-analysis.md) - Bibliographic and statistical analysis
- [Guides AGENTS.md](AGENTS.md) - Guide organization and standards

