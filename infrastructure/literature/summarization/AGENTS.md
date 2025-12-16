# Summarization Module - Documentation

## Purpose

The Summarization module provides paper summarization with multi-stage generation, quality validation, and context extraction. It uses PDF text processing with section prioritization, structured context extraction, and automatic refinement based on validation feedback.

## Architecture

This module follows the **thin orchestrator pattern** with clear separation of concerns:

### Core Components

1. **SummarizationEngine** (`core.py`) - Main interface for paper summarization
   - Orchestrates summarization workflow
   - PDF text extraction with prioritization
   - Context extraction and structuring
   - Multi-stage summary generation (draft + refine)
   - Quality validation
   - Result management
   - Progress reporting

2. **MultiStageSummarizer** (`multi_stage_summarizer.py`) - Two-stage summarization process
   - Draft generation using structured context
   - Quality validation with detailed feedback
   - Automatic refinement addressing validation issues
   - Progress event emission

3. **ContextExtractor** (`context_extractor.py`) - Structured context extraction
   - Identifies key sections (abstract, intro, conclusion)
   - Extracts key terms from title and abstract
   - Creates structured context objects
   - Section detection and parsing

4. **PDFProcessor** (`pdf_processor.py`) - PDF text processing
   - Section prioritization (title, abstract, intro, conclusion)
   - Truncation with critical section preservation
   - Section detection and mapping
   - Character limit management

5. **SummarizationPromptBuilder** (`prompt_builder.py`) - Prompt generation
   - Draft generation prompts with examples
   - Refinement prompts with specific issue guidance
   - Context-aware prompt construction
   - Quality checklist integration
   - Requirements: 10-15 quotes, 1000-1500 words
   - Coverage requirements (methodology, results, discussion)

6. **Streaming Support** (`streaming.py`) - LLM generation with progress
   - Streaming wrapper with periodic progress updates
   - Chunk accumulation and word counting
   - Progress events every 5 seconds during generation
   - Feedback on chars/words received

7. **SummaryQualityValidator** (`validator.py`) - Quality validation
   - Length validation
   - Title matching
   - Pattern-based hallucination detection (AI language, code snippets)
   - Repetition analysis (sentence similarity scoring)
   - Quote presence validation
   - Detailed error reporting
   - **Note**: Term-based validation removed to reduce false positives

8. **PDFChunker** (`chunker.py`) - PDF text chunking
   - Preserves section boundaries when chunking
   - Prioritizes key sections (title, abstract, intro, conclusion)
   - Supports two-stage summarization for large documents
   - Handles section detection and mapping

9. **TextExtractor** (`extractor.py`) - Text extraction from PDFs
   - Extracts text from PDF files
   - Saves extracted text to `data/extracted_text/` directory
   - Handles first stage of summarization pipeline (PDF → extracted text)
   - Integrates with PDFProcessor for text extraction

10. **Utility Functions** (`utils.py`) - Shared utilities
    - `detect_model_size()` - Detects LLM model size from client or metadata
    - Model size detection from multiple sources (client config, metadata, env vars)
    - Returns model size in billions of parameters for model-aware configuration

11. **Progress Tracking** - Progress reporting
    - Stage-level progress events
    - Timing and metadata tracking
    - Integration with ProgressTracker
    - Console output

### Architecture Diagram

```
SummarizationEngine (core.py)
├── PDFProcessor (pdf_processor.py)
│   └── PrioritizedPDFText extraction
├── ContextExtractor (context_extractor.py)
│   └── SummarizationContext creation
├── MultiStageSummarizer (multi_stage_summarizer.py)
│   ├── Draft generation
│   ├── Quality validation
│   └── Refinement (with retries)
├── SummarizationPromptBuilder (prompt_builder.py)
│   ├── Draft prompts
│   └── Refinement prompts
└── SummaryQualityValidator (validator.py)
    └── Comprehensive validation

Progress Events (models.py)
└── SummarizationProgressEvent
    ├── Stage tracking
    ├── Status updates
    └── Metadata
```

### Module Files

| File | Purpose |
|------|---------|
| `__init__.py` | Public API exports |
| `core.py` | Main `SummarizationEngine` class |
| `multi_stage_summarizer.py` | Two-stage summarization (draft + refine) |
| `context_extractor.py` | Structured context extraction |
| `pdf_processor.py` | PDF text processing with prioritization |
| `prompt_builder.py` | Enhanced prompt generation |
| `streaming.py` | Streaming wrapper with progress updates |
| `validator.py` | Quality validation |
| `models.py` | Data models (Result, Context, ValidationResult, ProgressEvent) |
| `metadata.py` | Summary metadata management |
| `parser.py` | Summary parsing utilities |
| `orchestrator.py` | Workflow orchestration function |
| `chunker.py` | PDF text chunking for two-stage summarization |
| `extractor.py` | Text extraction from PDFs for summarization pipeline |
| `utils.py` | Utility functions for model detection and configuration |
| `README.md` | Quick reference guide |
| `AGENTS.md` | Documentation |

## Multi-Stage Workflow

The summarization process follows a multi-stage approach:

### Stage 1: PDF Text Extraction
- Extract text from PDF using prioritized processor
- Preserve critical sections (title, abstract, intro, conclusion)
- Apply character limits while maintaining structure
- **Progress Event**: `pdf_extraction` (started/completed)

### Stage 2: Context Extraction
- Identify and extract key sections
- Extract key terms from title and abstract
- Create structured `SummarizationContext` object
- **Progress Event**: `context_extraction` (started/completed)

### Stage 3: Draft Generation
- Build optimized, model-size-aware prompt (~50% shorter for small models):
  - 8-12 direct quotes minimum
  - 800-1200 words coverage
  - Methodology, results, and discussion sections required
  - Explicit anti-repetition instructions
- Generate initial draft summary using LLM with **streaming support** and **model-aware generation options**
- Apply **post-processing deduplication** before validation
- Progress updates every 5 seconds showing chars/words received
- **Progress Event**: `draft_generation` (started/in_progress/completed)

### Stage 4: Quality Validation
- Validate summary quality
- Check for errors, warnings, missing elements
- Calculate quality score (0.0 to 1.0)
- **Progress Event**: `validation` (started/completed)

### Stage 5: Refinement (if needed)
- If validation score < 0.5 or errors found:
  - Build refinement prompt addressing specific issues
  - Use **fallback strategies** on later attempts (simpler prompts, lower temperature)
  - Generate refined summary with model-aware options
  - Apply **post-processing deduplication** before validation
  - Re-validate refined summary
  - Repeat up to `max_refinement_attempts` (default: 2)
- **Progress Event**: `refinement` (started/completed, per attempt)

### Stage 6: Pass 2 - Claims and Quotes Extraction
- Extract key claims and hypotheses from the paper
- Identify important direct quotes
- Generate markdown-formatted claims and quotes document
- **Progress Event**: `claims_extraction` (started/completed)

### Stage 7: Pass 3 - Methods and Tools Analysis
- Analyze algorithms and methodologies used
- Extract software frameworks and libraries
- Identify datasets and evaluation metrics
- Document software tools and platforms
- **Validate output quality**: Detect bloat, hallucinations, and formatting issues
- **Apply post-processing**: Remove "Not specified in paper" repetition, validate against source PDF
- **Quality checks**: Output size limits, hallucinated tools detection, formatting fixes
- Generate markdown-formatted methods and tools document
- **Progress Event**: `methods_analysis` (started/completed)

### Stage 8: Pass 4 - Paper Classification
- Classify paper by research type using full paper context
- Categories: Core/Theory/Math, Translation/Tool Development, or Applied
- For Applied papers, identify specific domain (e.g., Computer Science, Biology)
- Generate structured JSON classification with confidence score
- Store classification in library entry metadata
- **Progress Event**: `classification` (started/completed)

### Stage 9: Result Management
- Accept or reject final summary based on validation
- Save summary to markdown file (includes classification section)
- Save claims/quotes and methods/tools to separate files
- Save metadata to JSON
- Update library entry with classification
- **Progress Event**: `completed` (success/failed)

## Progress Tracking

### Progress Events

The module emits progress events at each stage:

```python
@dataclass
class SummarizationProgressEvent:
    citation_key: str
    stage: str  # "pdf_extraction", "context_extraction", "draft_generation", "validation", "refinement", "claims_extraction", "methods_analysis", "classification"
    status: str  # "started", "completed", "failed"
    message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
```

### Progress Callback Interface

```python
def progress_callback(event: SummarizationProgressEvent) -> None:
    """Handle progress event.
    
    Args:
        event: Progress event with stage, status, and metadata.
    """
    if event.status == "started":
        logger.info(f"[{event.citation_key}] {event.stage}: {event.message}")
    elif event.status == "in_progress":
        # Handle streaming progress updates
        if event.metadata.get("streaming"):
            chars = event.metadata.get("chars_received", 0)
            words = event.metadata.get("words_received", 0)
            elapsed = event.metadata.get("elapsed_time", 0)
            logger.info(
                f"[{event.citation_key}] ↻ {event.stage}: "
                f"Streaming: {chars:,} chars, {words:,} words ({elapsed:.1f}s)"
            )
        else:
            logger.info(f"[{event.citation_key}] ↻ {event.stage}: {event.message}")
    elif event.status == "completed":
        logger.info(f"[{event.citation_key}] ✓ {event.stage} completed: {event.message}")
```

### Integration with ProgressTracker

The workflow integrates progress events with `ProgressTracker`:

- Paper-level status updates (pending → processing → summarized/failed)
- Stage-level progress logging
- Console output with formatted progress
- Progress bar updates

### Progress Display Format

During summarization, progress is displayed as:

```
[1/9] Processing pazem2025free...
  ✓ PDF extraction (2.3s, 116,726 chars)
  ✓ Context extraction (0.01s, abstract=1703 chars, intro=2999 chars)
  → Draft generation: Generating draft summary...
  ↻ Draft generation: Streaming: 5,234 chars, 756 words (5.2s elapsed)
  ↻ Draft generation: Streaming: 12,456 chars, 1,823 words (10.4s elapsed)
  ↻ Draft generation: Streaming: 18,923 chars, 2,756 words (15.6s elapsed)
  ✓ Draft generation: Draft generated: 2,156 words (45.2s)
  → Validation: Validating summary quality...
  ✓ Validation completed (3.1s, score: 0.85)
  → Refinement: Refining summary (attempt 1/2)...
  ↻ Refinement: Streaming: 8,234 chars, 1,156 words (5.1s elapsed)
  ↻ Refinement: Streaming: 15,678 chars, 2,234 words (10.3s elapsed)
  ✓ Refinement: Refinement completed: 2,456 words (38.5s)
  ✓ Summary accepted (score: 0.92, total: 89.1s)
```

**Streaming Progress Updates:**
- Updates every 5 seconds during LLM generation
- Shows accumulated characters, words, and elapsed time
- Provides feedback during long generation times (60-80+ seconds)

## Usage Examples

### Basic Usage

```python
from infrastructure.literature.summarization import SummarizationEngine
from infrastructure.literature.sources import SearchResult
from infrastructure.llm import LLMClient
from pathlib import Path

# Initialize LLM client
llm_client = LLMClient()

# Initialize engine
engine = SummarizationEngine(llm_client)

# Summarize paper
result = engine.summarize_paper(
    result=search_result,
    pdf_path=Path("data/pdfs/paper.pdf"),
    max_retries=2
)

# Check result (summary_text is always present, even if validation failed)
if result.summary_text:
    print(f"Summary generated: {result.output_words} words")
    print(f"Quality score: {result.quality_score:.2f}")
    print(f"Validation status: {'Accepted' if result.success else 'Rejected'}")
    if result.validation_errors:
        print(f"Validation errors: {len(result.validation_errors)}")
    
    # Save summary (always saves, even if validation failed)
    summary_path = engine.save_summary(
        result=search_result,
        summary_result=result,
        output_dir=Path("data/summaries"),
        pdf_path=Path("data/pdfs/paper.pdf")
    )
    print(f"Summary saved to: {summary_path}")
    # Saved file includes validation metadata (status, score, errors)
else:
    print(f"Summarization failed: {result.error}")
```

### With Progress Callback

```python
from infrastructure.literature.summarization import (
    SummarizationEngine,
    SummarizationProgressEvent
)

def progress_callback(event: SummarizationProgressEvent):
    if event.status == "started":
        print(f"→ {event.stage}: {event.message}")
    elif event.status == "completed":
        print(f"✓ {event.stage} completed: {event.message}")

engine = SummarizationEngine(llm_client)

result = engine.summarize_paper(
    result=search_result,
    pdf_path=Path("data/pdfs/paper.pdf"),
    progress_callback=progress_callback
)
```

### Using with LiteratureWorkflow

```python
from infrastructure.literature import LiteratureWorkflow, LiteratureSearch
from infrastructure.literature.summarization import SummarizationEngine
from infrastructure.llm import LLMClient

# Initialize components
llm_client = LLMClient()
literature_search = LiteratureSearch()
summarizer = SummarizationEngine(llm_client)

# Create workflow
workflow = LiteratureWorkflow(literature_search)
workflow.set_summarizer(summarizer)

# Execute complete workflow
result = workflow.execute_search_and_summarize(
    keywords=["machine learning", "neural networks"],
    limit_per_keyword=10,
    max_parallel_summaries=2
)
```

### Using Orchestrator Function

```python
from infrastructure.literature.workflow import LiteratureWorkflow
from infrastructure.literature.summarization.orchestrator import run_summarize

# Create and configure workflow
workflow = LiteratureWorkflow(literature_search)
workflow.set_summarizer(summarizer)
workflow.set_progress_tracker(progress_tracker)

# Run summarization for papers with PDFs
exit_code = run_summarize(workflow)
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MAX_INPUT_LENGTH` | Model-aware | Maximum PDF characters (50K for <7B, 100K for 7-13B, 200K for >13B) |
| `MAX_PARALLEL_SUMMARIES` | `1` | Maximum parallel summarization workers |
| `LITERATURE_TWO_STAGE_ENABLED` | `true` | Enable two-stage mode for large texts (>200K chars) |
| `LITERATURE_CHUNK_SIZE` | `15000` | Target chunk size for two-stage mode |
| `LITERATURE_CHUNK_OVERLAP` | Model-aware | Chunk overlap (200 for <7B, 500 for >=7B) |
| `LITERATURE_TWO_STAGE_THRESHOLD` | `200000` | Text size threshold to trigger two-stage mode |

### Engine Configuration

```python
from infrastructure.literature.summarization.prompt_builder import SummarizationPromptBuilder

# Prompt builder automatically detects model size from LLM client
builder = SummarizationPromptBuilder(llm_client=llm_client)

engine = SummarizationEngine(
    llm_client=llm_client,
    quality_validator=validator,  # Optional: custom validator
    context_extractor=extractor,   # Optional: custom extractor
    prompt_builder=builder,        # Optional: custom builder (auto-detects model size)
    max_pdf_chars=None,           # Optional: None = auto-detect from model size
    config=None                   # Optional: LiteratureConfig (created from env if None)
)
```

**Model-Aware Configuration:**
- Automatically detects model size from LLM client configuration
- Adjusts temperature: 0.3 (<7B), 0.4 (7-13B), 0.5 (>13B)
- Adjusts max_tokens: 2000-3000 based on model size and stage
- Adjusts PDF char limit: 50K (<7B), 100K (7-13B), 200K (>13B)
- Adjusts chunk overlap: 200 (<7B), 500 (>=7B)

### Multi-Stage Summarizer Configuration

```python
from infrastructure.literature.summarization.multi_stage_summarizer import MultiStageSummarizer

summarizer = MultiStageSummarizer(
    llm_client=llm_client,
    validator=validator,
    prompt_builder=builder,
    max_refinement_attempts=2  # Maximum refinement attempts
)
```

## Data Models

### SummarizationResult

```python
@dataclass
class SummarizationResult:
    citation_key: str
    success: bool  # True if validation passed (score >= 0.5)
    summary_text: Optional[str] = None  # Always present if generation succeeded (even if validation failed)
    input_chars: int = 0
    input_words: int = 0
    output_words: int = 0
    generation_time: float = 0.0
    attempts: int = 0
    error: Optional[str] = None
    quality_score: float = 0.0  # Validation score (0.0 to 1.0)
    validation_errors: List[str] = field(default_factory=list)  # Validation issues (if any)
    summary_path: Optional[Path] = None
    skipped: bool = False
    
    # Note: summary_text is always included when generation succeeds,
    # regardless of validation status. This ensures summaries can be
    # saved for review even if validation fails.
    
    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio (output/input words)."""
        return self.output_words / max(1, self.input_words)
    
    @property
    def words_per_second(self) -> float:
        """Calculate generation speed in words per second."""
        return self.output_words / max(0.001, self.generation_time)
    
    classification: Optional[PaperClassification] = None  # Paper classification result
```

### PaperClassification

```python
@dataclass
class PaperClassification:
    """Paper classification result from Pass 4.
    
    Attributes:
        category: Primary category ("core_theory_math", "translation_tool", "applied")
        domain: Domain name if category is "applied" (e.g., "Computer Science", "Biology")
        confidence: Confidence score (0.0 to 1.0)
        reasoning: Brief explanation of classification
    """
    category: str  # "core_theory_math", "translation_tool", "applied"
    domain: Optional[str] = None  # Required if category is "applied"
    confidence: float = 0.0
    reasoning: Optional[str] = None
```

### SummarizationContext

```python
@dataclass
class SummarizationContext:
    title: str
    abstract: str
    introduction: str
    conclusion: str
    key_terms: List[str]
    equations: List[str]
    methods: Optional[str] = None
    results: Optional[str] = None
    discussion: Optional[str] = None
    full_text: str = ""
```

### ValidationResult

```python
@dataclass
class ValidationResult:
    is_valid: bool
    score: float  # 0.0 to 1.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_key_terms: List[str] = field(default_factory=list)
    quote_count: int = 0
    repetition_issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def has_hard_failure(self) -> bool:
        """Check if validation indicates hard failure (should reject immediately)."""
        # Returns True for title mismatch, major hallucination, severe repetition
```

### SummarizationProgressEvent

```python
@dataclass
class SummarizationProgressEvent:
    citation_key: str
    stage: str  # "pdf_extraction", "context_extraction", "draft_generation", "validation", "refinement", "claims_extraction", "methods_analysis", "classification"
    status: str  # "started", "completed", "failed"
    message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
```

## Key Features

### Multi-Stage Generation
- Draft generation using structured context
- Quality validation with detailed feedback
- Automatic refinement addressing specific issues
- Up to 2 refinement attempts (configurable)

### PDF Processing
- Section prioritization (title, abstract, intro, conclusion preserved)
- Character limit management with structure preservation
- Section detection and mapping
- Truncation with critical section preservation

### Quality Validation
- Quality checks
- Hard failure detection (title mismatch, hallucination, severe repetition)
- Quality scoring (0.0 to 1.0)
- Detailed error and warning reporting
- Refinement guidance generation

### Progress
- Stage-level progress events
- Console output
- Progress bar integration
- Timing and metadata tracking

### Context Extraction
- Key section identification
- Key term extraction
- Structured context creation
- Equation detection

## Best Practices

### 1. Use Progress Callbacks
Always provide progress callbacks for long-running operations:

```python
def progress_callback(event: SummarizationProgressEvent):
    logger.info(f"[{event.citation_key}] {event.stage}: {event.message}")

engine.summarize_paper(
    result=search_result,
    pdf_path=pdf_path,
    progress_callback=progress_callback
)
```

### 2. Handle Validation Results
Check validation results and handle failures appropriately:

```python
if result.success:
    if result.quality_score >= 0.7:
        # High quality summary
        save_summary(result)
    elif result.quality_score >= 0.5:
        # Acceptable quality, may need review
        save_summary(result, flag_for_review=True)
    else:
        # Low quality, consider retry
        logger.warning(f"Low quality summary: {result.quality_score}")
```

### 3. Configure Character Limits
Adjust `max_pdf_chars` based on your LLM's context window:

```python
# For models with large context windows
engine = SummarizationEngine(
    llm_client=llm_client,
    max_pdf_chars=500000  # 500K characters
)

# For models with smaller context windows
engine = SummarizationEngine(
    llm_client=llm_client,
    max_pdf_chars=100000  # 100K characters
)
```

### 4. Parallel Processing
Use parallel processing for multiple papers:

```python
workflow.execute_search_and_summarize(
    keywords=["machine learning"],
    max_parallel_summaries=4  # Process 4 papers in parallel
)
```

### 5. Error Handling
Always check for errors and handle them gracefully:

```python
result = engine.summarize_paper(result, pdf_path)

if not result.success:
    logger.error(f"Summarization failed: {result.error}")
    if "LLM connection" in result.error:
        # Retry with backoff
        retry_with_backoff(...)
    elif "PDF extraction" in result.error:
        # Check PDF file
        validate_pdf(pdf_path)
```

## Troubleshooting

### Low Quality Scores
- **Issue**: Summaries consistently score below 0.5
- **Solutions**:
  - Check PDF text extraction quality
  - Verify context extraction is finding key sections
  - Review validation errors for specific issues
  - Consider adjusting refinement attempts

### Slow Generation
- **Issue**: Summarization takes too long
- **Solutions**:
  - Reduce `max_pdf_chars` to process less text
  - Use parallel processing for multiple papers
  - Check LLM connection and model performance
  - Review progress events to identify bottlenecks

### Missing Sections
- **Issue**: Important sections not included in summary
- **Solutions**:
  - Check PDF text extraction (may be missing sections)
  - Verify context extraction is finding all sections
  - Review section detection patterns
  - Increase `max_pdf_chars` if truncation is occurring

### Validation Errors
- **Issue**: Frequent validation errors
- **Solutions**:
  - Review validation error messages in saved summary files
  - Check for title mismatches (may indicate wrong PDF)
  - Note: Term-based validation has been removed to reduce false positives
  - Validation now focuses on structural quality (repetition, length, quotes)
  - Summaries are always saved with validation metadata for review
  - Consider adjusting validation thresholds if needed

### Progress Not Updating
- **Issue**: Progress events not appearing
- **Solutions**:
  - Verify progress callback is provided
  - Check that callback is being called
  - Review logging configuration
  - Ensure ProgressTracker is properly initialized

## Quality Standards and Output Validation

### Pass 3 Validation (Methods/Tools Analysis)

The system now includes comprehensive validation for methods/tools extraction:

**Validation Checks:**
- **Bloat Detection**: Identifies excessive "Not specified in paper" repetition (>3 instances)
- **Hallucination Detection**: Validates extracted tools exist in source PDF text
- **Size Limits**: Enforces maximum 200 lines and 1500 words per output
- **Formatting**: Fixes spacing and punctuation issues
- **Source Validation**: Only allows explicitly mentioned tools/frameworks

**Validation Process:**
1. **Pre-validation**: Check for bloat patterns before source validation
2. **Source verification**: Fuzzy matching against PDF text with fallback strategies
3. **Post-processing**: Consolidate empty sections, remove artifacts
4. **Quality reporting**: Log validation warnings and corrections applied

**Common Issues Detected:**
- **Bloat**: 932+ repetitions of "Not specified in paper" (lapkovskis2025benchmarking regression)
- **Hallucinations**: Tools like ImageNet, CIFAR-10, Google Colab, AWS not in paper text
- **Size anomalies**: Files exceeding 990 lines due to repetition

### Summary Quality Standards

**Accepted Quality Criteria:**
- **Information Density**: ≥0.3 specific indicators per 50 words (numbers, metrics, methods)
- **Repetition Control**: No phrase repeated >3 times, no severe sentence-level repetition
- **Evidence Support**: Direct quotes or specific references for key claims
- **Length Appropriateness**: 600-1000 words for comprehensive coverage
- **Title Accuracy**: Summary title matches paper title exactly

**Quality Metrics Tracked:**
- **Success Rate**: Percentage of operations meeting quality threshold (≥0.5 score)
- **Hard Failure Rate**: Operations with critical issues (title mismatch, severe repetition)
- **Retry Rate**: Average number of refinement attempts needed
- **Information Density**: Ratio of specific details to total content
- **Validation Patterns**: Most common errors and warnings across operations

### Common Failure Modes

**Methods/Tools Bloat:**
- **Symptoms**: Output files >500 lines with repetitive "Not specified in paper"
- **Root Cause**: LLM instructed to list items even when not found
- **Prevention**: Template anti-bloat instructions, post-validation consolidation

**Hallucinated Tools:**
- **Symptoms**: Tools/frameworks listed that don't appear in paper text
- **Root Cause**: Speculative tool inference based on methodology
- **Prevention**: Strict source validation, explicit mention requirements

**Quote Quality Issues:**
- **Symptoms**: Missing quote text, meta-commentary in quotes, duplicate quotes
- **Root Cause**: Poor extraction formatting, inadequate deduplication
- **Prevention**: Enhanced quote validation, formatting checks, duplicate detection

**Low Information Density:**
- **Symptoms**: Generic phrases ("high performance", "effective") repeated excessively
- **Root Cause**: LLM generating vague content without specific details
- **Prevention**: Information density validation, specificity requirements

**Severe Repetition:**
- **Symptoms**: Same sentences/paragraphs repeated multiple times
- **Root Cause**: LLM getting stuck in repetitive patterns
- **Prevention**: Aggressive deduplication, temperature adjustments, refinement strategies

### Quality Monitoring

**Real-time Metrics:**
- **Quality Tracker**: Aggregates metrics across all summarization operations
- **Health Status**: System health assessment (excellent/good/fair/poor)
- **Trend Analysis**: Quality score trends over time (last 100 operations)
- **Failure Analysis**: Most common validation errors and warnings

**Automated Quality Gates:**
- **Pre-save validation**: All outputs validated before file creation
- **Anomaly detection**: Size and quality thresholds prevent problematic outputs
- **Regression prevention**: Known failure patterns automatically detected and corrected

## See Also

- [`README.md`](README.md) - Quick reference
- [`../AGENTS.md`](../AGENTS.md) - Literature module overview
- [`../../llm/AGENTS.md`](../../llm/AGENTS.md) - LLM integration documentation
- [`../workflow/AGENTS.md`](../workflow/AGENTS.md) - Workflow orchestration

## Recent Improvements

### Term-Based Validation Removal
- **Removed**: Term-based topic matching and hallucination detection
- **Reason**: Produced too many false positives (e.g., flagging "posits", "reflecting" as hallucinations)
- **Result**: Validation now focuses on structural quality (repetition, length, pattern-based hallucination)
- **Impact**: Reduced false rejection rate, more reliable validation

### Always-Save Behavior
- **Change**: Summaries are always saved, even when validation fails
- **Implementation**: `summary_text` is always included in `SummarizationResult` when generation succeeds
- **Benefit**: No work is lost; rejected summaries can be reviewed and manually assessed
- **Metadata**: Saved files include validation status, quality score, and error list

### Model-Aware Configuration
- **Automatic Detection**: Model size detected from LLM client configuration
- **Temperature**: 0.3 (<7B), 0.4 (7-13B), 0.5 (>13B)
- **Max Tokens**: 2000-3000 based on model size and stage
- **PDF Limits**: 50K (<7B), 100K (7-13B), 200K (>13B)
- **Chunk Overlap**: 200 (<7B), 500 (>=7B)

### Optimized Prompts
- **Reduction**: ~50% shorter prompts while maintaining clarity
- **Model-Size Aware**: Shorter prompts for small models (<7B)
- **Consolidated**: Removed redundant instructions
- **Anti-Repetition**: Explicit guidance to avoid repetition

### Post-Processing Deduplication
- **Applied**: After draft generation and after each refinement attempt
- **Method**: Aggressive deduplication using sentence similarity (0.85 threshold)
- **Benefit**: Reduces repetition before validation, improving quality scores

### Fallback Refinement Strategies
- **Strategy 1**: Standard refinement with full prompt
- **Strategy 2+**: Simplified prompts with lower temperature
- **Benefit**: Better success rate when initial refinement fails

## API Reference

### SummarizationEngine

**Main Methods:**
- `summarize_paper(result, pdf_path, max_retries=2, progress_callback=None)` - Generate summary
  - Returns `SummarizationResult` with `summary_text` always present (if generation succeeded)
  - Validation status available in `success`, `quality_score`, and `validation_errors`
- `save_summary(result, summary_result, output_dir, pdf_path=None)` - Save summary to file
  - Always saves if `summary_text` exists (regardless of validation status)
  - Includes validation metadata in saved file

**Properties:**
- `quality_validator` - Quality validator instance
- `max_pdf_chars` - Maximum PDF characters to process
- `config` - Literature configuration (used for library index updates)

### MultiStageSummarizer

**Main Methods:**
- `summarize_with_refinement(context, pdf_text, metadata, citation_key, progress_callback=None)` - Generate with refinement
- `generate_draft(context, metadata, progress_callback=None)` - Generate initial draft
- `refine_summary(draft, issues, context, citation_key, progress_callback=None)` - Refine summary
- `validate_and_accept(summary, context, pdf_text, paper_title, citation_key)` - Validate summary

### ContextExtractor

**Main Methods:**
- `create_summarization_context(pdf_text, title, max_chars=None)` - Create structured context
- `extract_key_terms(title, abstract)` - Extract key terms
- `identify_sections(pdf_text)` - Identify paper sections

### PDFProcessor

**Main Methods:**
- `extract_prioritized_text(pdf_path, max_chars)` - Extract text with prioritization
- `identify_sections(pdf_text)` - Identify sections in PDF text

### SummaryQualityValidator

**Main Methods:**
- `validate_summary_detailed(summary, pdf_text, citation_key, paper_title, key_terms)` - Detailed validation
- `validate_summary(summary, pdf_text, citation_key, paper_title=None)` - Basic validation

### PDFChunker

**Main Methods:**
- `chunk_text(text, chunk_size, overlap)` - Chunk text preserving section boundaries
- `chunk_with_section_awareness(text, sections, chunk_size, overlap)` - Chunk with section awareness

### TextExtractor

**Main Methods:**
- `extract_and_save(pdf_path, citation_key)` - Extract text from PDF and save to extracted_text directory
- `extract_text(pdf_path)` - Extract text from PDF file

### Utility Functions

**Functions:**
- `detect_model_size(llm_client, metadata)` - Detect model size in billions of parameters
  - Attempts detection from LLM client configuration, metadata, or environment variables
  - Returns model size (e.g., 4.0 for gemma3:4b) or 7.0 as default

## Examples

See `README.md` for quick reference and usage examples.
