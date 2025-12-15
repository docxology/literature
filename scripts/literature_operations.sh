#!/usr/bin/env bash
################################################################################
# Literature Operations Module
#
# Literature operation functions for executing workflows:
# - Full pipeline (search → download → extract → summarize)
# - Meta-analysis pipeline
# - Individual operations (search, download, extract, summarize, cleanup, LLM)
# - Operation logging helpers
#
# Dependencies:
#   - bash_utils.sh (must be sourced before this module)
#   - literature_helpers.sh (must be sourced before this module)
#   - literature_errors.sh (must be sourced before this module)
#   - REPO_ROOT, PYTHON_SCRIPT constants (must be defined)
################################################################################

# ============================================================================
# Operation Logging Helpers
# ============================================================================

# Function: log_operation_start
# Purpose: Log operation start with context and parameters
# Args:
#   $1: Operation name
#   $2: Operation description (optional)
#   $3: Additional parameters (optional)
# Side effects: Logs to both terminal and file
log_operation_start() {
    local op_name="${1:-unknown}"
    local op_desc="${2:-}"
    local params="${3:-}"
    
    local message="Starting operation: ${op_name}"
    if [[ -n "$op_desc" ]]; then
        message="${message} - ${op_desc}"
    fi
    if [[ -n "$params" ]]; then
        message="${message} (${params})"
    fi
    
    log_info_to_file "$message"
    log_info "$message"
    log_debug_to_file "Operation start: ${op_name} at $(date '+%Y-%m-%d %H:%M:%S')"
}

# Function: log_operation_end
# Purpose: Log operation end with metrics
# Args:
#   $1: Operation name
#   $2: Exit code (0 for success)
#   $3: Duration in seconds
#   $4: Additional metrics (optional, format: "key=value,key2=value2")
# Side effects: Logs to both terminal and file
log_operation_end() {
    local op_name="${1:-unknown}"
    local exit_code="${2:-0}"
    local duration="${3:-0}"
    local metrics="${4:-}"
    
    local status="completed"
    local status_color="$GREEN"
    local status_symbol="✓"
    
    if [[ $exit_code -ne 0 ]]; then
        status="failed"
        status_color="$RED"
        status_symbol="✗"
    fi
    
    local message="${status_symbol} Operation ${status}: ${op_name}"
    if validate_numeric "$duration" 0; then
        message="${message} (duration: $(format_duration "$duration"))"
    fi
    if [[ -n "$metrics" ]]; then
        message="${message} [${metrics}]"
    fi
    
    if [[ $exit_code -eq 0 ]]; then
        log_success_to_file "$message"
        echo -e "${status_color}${message}${NC}"
    else
        log_error_to_file "$message (exit code: ${exit_code})"
        echo -e "${status_color}${message}${NC}"
    fi
    
    log_debug_to_file "Operation end: ${op_name} at $(date '+%Y-%m-%d %H:%M:%S')"
}

# ============================================================================
# Environment Setup Operations
# ============================================================================

# Function: run_environment_check_dependencies
# Purpose: Check dependencies (Python, Ollama, required packages)
# Args: None
# Returns:
#   0: All dependencies available
#   1: Missing dependencies
# Side effects:
#   - Checks Python version and availability
#   - Checks Ollama installation and running status
#   - Checks required Python packages
# Dependencies:
#   - check_ollama_running()
# Example:
#   run_environment_check_dependencies
run_environment_check_dependencies() {
    log_header_to_file "CHECK DEPENDENCIES"
    log_header "CHECK DEPENDENCIES"
    
    local errors=0
    local warnings=0
    
    echo
    log_info "Checking dependencies..."
    echo
    
    # Check Python
    log_info "Python:"
    if command -v python3 &> /dev/null; then
        local python_version
        python_version=$(python3 --version 2>&1 || echo "unknown")
        log_success "  ✓ Python3 found: $python_version"
    else
        log_error "  ✗ Python3 not found in PATH"
        ((errors++))
    fi
    
    # Check required Python packages
    log_info "Python packages:"
    local required_packages=("infrastructure" "pytest" "numpy" "matplotlib")
    for package in "${required_packages[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            log_success "  ✓ $package installed"
        else
            log_warning "  ✗ $package not found"
            ((warnings++))
        fi
    done
    
    # Check Ollama
    log_info "Ollama:"
    if command -v ollama &> /dev/null; then
        log_success "  ✓ Ollama command found"
        local ollama_available
        ollama_available=$(check_ollama_running)
        if [[ "$ollama_available" == "true" ]]; then
            log_success "  ✓ Ollama server is running"
        else
            log_warning "  ⚠ Ollama installed but server not running"
            log_info "    Start with: ollama serve"
            ((warnings++))
        fi
    else
        log_warning "  ⚠ Ollama not found (required for LLM operations)"
        log_info "    Install from: https://ollama.ai"
        ((warnings++))
    fi
    
    echo
    if [[ $errors -eq 0 ]] && [[ $warnings -eq 0 ]]; then
        log_success "All dependencies available"
        return 0
    elif [[ $errors -eq 0 ]]; then
        log_warning "Dependencies check completed with $warnings warning(s)"
        return 0
    else
        log_error "Dependencies check failed with $errors error(s) and $warnings warning(s)"
        return 1
    fi
}

# Function: run_environment_validate_config
# Purpose: Validate configuration (env vars, file paths)
# Args: None
# Returns:
#   0: Configuration valid
#   1: Configuration issues found
# Side effects:
#   - Validates environment variables
#   - Checks file paths and permissions
# Dependencies:
#   - check_environment()
# Example:
#   run_environment_validate_config
run_environment_validate_config() {
    log_header_to_file "VALIDATE CONFIGURATION"
    log_header "VALIDATE CONFIGURATION"
    
    echo
    log_info "Validating configuration..."
    echo
    
    # Use existing check_environment function
    if check_environment; then
        log_success "Configuration validation passed"
        return 0
    else
        log_error "Configuration validation failed"
        return 1
    fi
}

# Function: run_environment_create_directories
# Purpose: Create data/ subdirectories
# Args: None
# Returns:
#   0: Directories created successfully
#   1: Failed to create directories
# Side effects:
#   - Creates data/ and subdirectories if they don't exist
# Example:
#   run_environment_create_directories
run_environment_create_directories() {
    log_header_to_file "CREATE DIRECTORIES"
    log_header "CREATE DIRECTORIES"
    
    if ! cd "$REPO_ROOT" 2>/dev/null; then
        log_error "Cannot change to repository root: $REPO_ROOT"
        return 1
    fi
    
    echo
    log_info "Creating data directories..."
    echo
    
    local directories=(
        "data"
        "data/pdfs"
        "data/summaries"
        "data/extracted_text"
        "data/output"
    )
    
    local created=0
    local existing=0
    local failed=0
    
    for dir in "${directories[@]}"; do
        if [[ -d "$dir" ]]; then
            log_info "  ✓ $dir (already exists)"
            ((existing++))
        else
            if mkdir -p "$dir" 2>/dev/null; then
                log_success "  ✓ Created $dir"
                ((created++))
            else
                log_error "  ✗ Failed to create $dir"
                ((failed++))
            fi
        fi
    done
    
    echo
    if [[ $failed -eq 0 ]]; then
        log_success "Directory setup complete: $created created, $existing existing"
        return 0
    else
        log_error "Directory setup failed: $failed error(s)"
        return 1
    fi
}

# Function: run_environment_check_connectivity
# Purpose: Check connectivity (network, Ollama server)
# Args: None
# Returns:
#   0: Connectivity OK
#   1: Connectivity issues
# Side effects:
#   - Tests network connectivity
#   - Checks Ollama server status
# Dependencies:
#   - check_ollama_running()
# Example:
#   run_environment_check_connectivity
run_environment_check_connectivity() {
    log_header_to_file "CHECK CONNECTIVITY"
    log_header "CHECK CONNECTIVITY"
    
    echo
    log_info "Checking connectivity..."
    echo
    
    local errors=0
    
    # Check network connectivity
    log_info "Network:"
    if ping -c 1 -W 2 8.8.8.8 &> /dev/null 2>&1 || ping -c 1 -W 2 google.com &> /dev/null 2>&1; then
        log_success "  ✓ Network connectivity OK"
    else
        log_warning "  ⚠ Network connectivity test failed (may be firewall/proxy issue)"
        log_info "    Some operations may require network access"
    fi
    
    # Check Ollama server
    log_info "Ollama server:"
    local ollama_available
    ollama_available=$(check_ollama_running)
    if [[ "$ollama_available" == "true" ]]; then
        log_success "  ✓ Ollama server is running"
    else
        log_warning "  ⚠ Ollama server is not running"
        log_info "    Start with: ollama serve"
        log_info "    Required for LLM-based analysis operations"
        ((errors++))
    fi
    
    echo
    if [[ $errors -eq 0 ]]; then
        log_success "Connectivity check passed"
        return 0
    else
        log_warning "Connectivity check completed with $errors issue(s)"
        return 0  # Return 0 as warnings don't block operations
    fi
}

# ============================================================================
# Individual Operations (by category)
# ============================================================================

# Function: run_literature_meta_analysis
# Purpose: Execute standard meta-analysis on existing library data (no embeddings)
# Args: None
# Returns:
#   0: Success
#   1: Failure
# Side effects:
#   - Executes Python script with --meta-analysis flag (without --with-embeddings)
#   - Analyzes existing citations, PDFs, and extracted text in library
#   - Does not search, download, or extract (use options 3.1 and 4.1-4.3 for those)
#   - Does not require Ollama
# Dependencies:
#   - check_python_script()
#   - run_python_script_with_retry()
# Example:
#   run_literature_meta_analysis
run_literature_meta_analysis() {
    log_header_to_file "META-ANALYSIS (STANDARD)"
    log_header "META-ANALYSIS (STANDARD)"
    
    if ! cd "$REPO_ROOT" 2>/dev/null; then
        log_error "Cannot change to repository root: $REPO_ROOT"
        return 1
    fi
    
    if ! check_python_script; then
        return 1
    fi
    
    echo
    log_info_to_file "🔄 Starting standard meta-analysis on existing library data..."
    log_info "🔄 Starting standard meta-analysis on existing library data..."
    echo
    log_info "This will analyze existing library data:"
    log_info "  • Bibliographic analysis (citations, venues, authors)"
    log_info "  • PCA analysis (if ≥2 papers with extracted text)"
    log_info "  • Keyword analysis (if abstracts available)"
    log_info "  • Temporal analysis (publication trends)"
    log_info "  • Metadata visualizations"
    echo
    log_info "Note: This does not search, download, or extract."
    log_info "      Use options 3.1 and 4.1-4.3 for those operations."
    echo
    
    local start_time=$(date +%s)
    log_debug "Pipeline start time: $start_time"
    
    # Use --meta-analysis without --with-embeddings for standard analysis
    # Clear options are handled interactively in the Python script
    if run_python_script_with_retry "--meta-analysis" 1; then
        local end_time=$(date +%s)
        local duration=$(get_elapsed_time "$start_time" "$end_time")
        echo
        log_success_to_file "✅ Standard meta-analysis pipeline complete in $(format_duration "$duration")"
        log_success "✅ Standard meta-analysis pipeline complete in $(format_duration "$duration")"
        echo
        log_info "📁 Output locations:"
        log_info "  • Bibliography: data/references.bib"
        log_info "  • JSON index: data/library.json"
        log_info "  • PDFs: data/pdfs/"
        log_info "  • Extracted text: data/extracted_text/"
        log_info "  • Visualizations: data/output/"
        echo
        return 0
    else
        log_error_to_file "❌ Standard meta-analysis pipeline failed"
        log_error "❌ Standard meta-analysis pipeline failed"
        return 1
    fi
}

# Function: run_literature_meta_analysis_with_embeddings
# Purpose: Execute full meta-analysis with Ollama embeddings on existing library data
# Args: None
# Returns:
#   0: Success
#   1: Failure
# Side effects:
#   - Executes Python script with --meta-analysis --with-embeddings flags
#   - Analyzes existing citations, PDFs, and extracted text in library
#   - Performs full meta-analysis including Ollama embedding analysis
#   - Does not search, download, or extract (use options 3.1 and 4.1-4.3 for those)
#   - Requires Ollama server running
# Dependencies:
#   - check_python_script()
#   - check_ollama_running()
#   - run_python_script_with_retry()
# Example:
#   run_literature_meta_analysis_with_embeddings
run_literature_meta_analysis_with_embeddings() {
    log_header_to_file "META-ANALYSIS (WITH EMBEDDINGS)"
    log_header "META-ANALYSIS (WITH EMBEDDINGS)"
    
    if ! cd "$REPO_ROOT" 2>/dev/null; then
        log_error "Cannot change to repository root: $REPO_ROOT"
        return 1
    fi
    
    if ! check_python_script; then
        return 1
    fi
    
    # Check Ollama availability (informational only)
    log_info "Checking Ollama availability for embedding analysis..."
    local ollama_available
    ollama_available=$(check_ollama_running)
    
    if [[ "$ollama_available" != "true" ]]; then
        log_warning "Ollama is not running - embedding analysis will be skipped"
        log_info "Standard meta-analysis will still run (PCA, keywords, authors, etc.)"
        log_info "To enable embedding analysis, start Ollama with: ollama serve"
        log_info "Then install an embedding model with: ollama pull embeddinggemma"
    else
        log_info "Ollama is available - embedding analysis will be included"
    fi
    
    echo
    log_info_to_file "🔄 Starting full meta-analysis with embeddings on existing library data..."
    log_info "🔄 Starting full meta-analysis with embeddings on existing library data..."
    echo
    log_info "This will analyze existing library data:"
    log_info "  • Standard meta-analysis (bibliographic, citations, PCA, keywords, authors, visualizations)"
    log_info "  • Ollama embedding analysis (semantic similarity, clustering, visualizations)"
    echo
    log_info "Note: This does not search, download, or extract."
    log_info "      Use options 3.1 and 4.1-4.3 for those operations."
    log_info "      Requires ≥2 papers with extracted text for embedding analysis."
    echo
    
    local start_time=$(date +%s)
    log_debug "Pipeline start time: $start_time"
    
    # Use --meta-analysis --with-embeddings for full analysis with embeddings
    # Clear options are handled interactively in the Python script
    if run_python_script_with_retry "--meta-analysis --with-embeddings" 1; then
        local end_time=$(date +%s)
        local duration=$(get_elapsed_time "$start_time" "$end_time")
        echo
        log_success_to_file "✅ Full meta-analysis pipeline with embeddings complete in $(format_duration "$duration")"
        log_success "✅ Full meta-analysis pipeline with embeddings complete in $(format_duration "$duration")"
        echo
        log_info "📁 Output locations:"
        log_info "  • Bibliography: data/references.bib"
        log_info "  • JSON index: data/library.json"
        log_info "  • PDFs: data/pdfs/"
        log_info "  • Extracted text: data/extracted_text/"
        log_info "  • Visualizations: data/output/"
        log_info "  • Embeddings: data/output/embeddings.json"
        log_info "  • Similarity matrix: data/output/embedding_similarity_matrix.csv"
        echo
        return 0
    else
        log_error_to_file "❌ Full meta-analysis pipeline with embeddings failed"
        log_error "❌ Full meta-analysis pipeline with embeddings failed"
        return 1
    fi
}

# ============================================================================
# Individual Operations
# ============================================================================

# Function: run_literature_search
# Purpose: Search literature and add papers to bibliography (network operation)
# Args: None
# Returns:
#   0: Success
#   1: Failure
# Side effects:
#   - Executes Python script with --search-only flag
#   - Updates bibliography and library files
#   - Requires network connection
# Dependencies:
#   - check_python_script()
#   - run_python_script_with_retry()
# Example:
#   run_literature_search
run_literature_search() {
    log_header_to_file "LITERATURE SEARCH (ADD TO BIBLIOGRAPHY)"
    log_header "LITERATURE SEARCH (ADD TO BIBLIOGRAPHY)"

    local operation_name="search"
    local start_time
    start_time=$(date +%s)
    
    log_operation_start "$operation_name" "Search literature and add to bibliography" "network only"

    if ! cd "$REPO_ROOT" 2>/dev/null; then
        log_error_with_context "Cannot change to repository root: $REPO_ROOT"
        log_operation_end "$operation_name" 1 0
        return 1
    fi

    if ! check_python_script; then
        log_operation_end "$operation_name" 1 0
        return 1
    fi

    local exit_code=0
    if run_python_script_with_retry "--search-only" 1; then
        exit_code=0
    else
        exit_code=$?
        log_info "Check network connection and API availability"
    fi
    
    local end_time
    end_time=$(date +%s)
    local duration
    duration=$(get_elapsed_time "$start_time" "$end_time")
    
    log_operation_end "$operation_name" "$exit_code" "$duration"
    return $exit_code
}

# Function: run_literature_download
# Purpose: Download PDFs for existing bibliography entries (network operation)
# Args: None
# Returns:
#   0: Success
#   1: Failure
# Side effects:
#   - Executes Python script with --download-only flag
#   - Downloads PDFs to data/pdfs/
#   - Requires network connection
# Dependencies:
#   - check_python_script()
#   - run_python_script_with_retry()
# Example:
#   run_literature_download
run_literature_download() {
    log_header_to_file "DOWNLOAD PDFs (FOR BIBLIOGRAPHY ENTRIES)"
    log_header "DOWNLOAD PDFs (FOR BIBLIOGRAPHY ENTRIES)"

    local operation_name="download"
    local start_time
    start_time=$(date +%s)
    
    log_operation_start "$operation_name" "Download PDFs for bibliography entries" "network only"

    if ! cd "$REPO_ROOT" 2>/dev/null; then
        log_error_with_context "Cannot change to repository root: $REPO_ROOT"
        log_operation_end "$operation_name" 1 0
        return 1
    fi

    if ! check_python_script; then
        log_operation_end "$operation_name" 1 0
        return 1
    fi

    local exit_code=0
    if run_python_script_with_retry "--download-only" 1; then
        exit_code=0
    else
        exit_code=$?
        log_info "Check network connection and PDF availability"
    fi
    
    local end_time
    end_time=$(date +%s)
    local duration
    duration=$(get_elapsed_time "$start_time" "$end_time")
    
    log_operation_end "$operation_name" "$exit_code" "$duration"
    return $exit_code
}

# Function: run_literature_extract_text
# Purpose: Extract text from PDFs in library (local operation)
# Args: None
# Returns:
#   0: Success
#   1: Failure
# Side effects:
#   - Executes Python script with --extract-text flag
#   - Creates text files in data/extracted_text/
#   - No network required
# Dependencies:
#   - check_python_script()
#   - run_python_script_with_retry()
# Example:
#   run_literature_extract_text
run_literature_extract_text() {
    log_header_to_file "EXTRACT TEXT FROM PDFs"
    log_header "EXTRACT TEXT FROM PDFs"

    if ! cd "$REPO_ROOT" 2>/dev/null; then
        log_error "Cannot change to repository root: $REPO_ROOT"
        return 1
    fi

    if ! check_python_script; then
        return 1
    fi

    log_info_to_file "Extracting text from PDFs (local operation, no network required)..."
    log_info "Extracting text from PDFs (local operation, no network required)..."
    
    if run_python_script_with_retry "--extract-text" 1; then
        log_success_to_file "Text extraction complete!"
        log_success "Text extraction complete!"
        return 0
    else
        log_error_to_file "Text extraction failed"
        log_error "Text extraction failed"
        log_info "Ensure PDFs exist in data/pdfs/ and have read permissions"
        return 1
    fi
}

# Function: run_literature_download_and_extract
# Purpose: Download PDFs and extract text (network then local)
# Args: None
# Returns:
#   0: Success
#   1: Failure
# Side effects:
#   - Downloads PDFs for bibliography entries
#   - Extracts text from downloaded PDFs
# Dependencies:
#   - run_literature_download()
#   - run_literature_extract_text()
# Example:
#   run_literature_download_and_extract
run_literature_download_and_extract() {
    log_header_to_file "DOWNLOAD PDFs AND EXTRACT TEXT"
    log_header "DOWNLOAD PDFs AND EXTRACT TEXT"
    
    local operation_name="download_and_extract"
    local start_time
    start_time=$(date +%s)
    
    log_operation_start "$operation_name" "Download PDFs then extract text"
    
    # First download
    if ! run_literature_download; then
        log_operation_end "$operation_name" 1 0
        return 1
    fi
    
    echo
    log_info "Proceeding to text extraction..."
    echo
    
    # Then extract
    if ! run_literature_extract_text; then
        log_operation_end "$operation_name" 1 0
        return 1
    fi
    
    local end_time
    end_time=$(date +%s)
    local duration
    duration=$(get_elapsed_time "$start_time" "$end_time")
    
    log_operation_end "$operation_name" 0 "$duration"
    return 0
}

# Function: run_literature_summarize
# Purpose: Generate AI-powered summaries for papers with PDFs (requires Ollama)
# Args: None
# Returns:
#   0: Success
#   1: Failure
# Side effects:
#   - Executes Python script with --summarize flag
#   - Creates summary files in data/summaries/
#   - Requires Ollama server running
# Dependencies:
#   - check_python_script()
#   - check_ollama_running()
#   - run_python_script_with_retry()
# Example:
#   run_literature_summarize
run_literature_summarize() {
    log_header_to_file "GENERATE SUMMARIES (FOR PAPERS WITH PDFs)"
    log_header "GENERATE SUMMARIES (FOR PAPERS WITH PDFs)"

    local operation_name="summarize"
    local start_time
    start_time=$(date +%s)
    
    log_operation_start "$operation_name" "Generate summaries for papers with PDFs" "requires Ollama"

    if ! cd "$REPO_ROOT" 2>/dev/null; then
        log_error_with_context "Cannot change to repository root: $REPO_ROOT"
        log_operation_end "$operation_name" 1 0
        return 1
    fi

    if ! check_python_script; then
        log_operation_end "$operation_name" 1 0
        return 1
    fi

    # Check Ollama availability
    log_info "Checking Ollama availability..."
    local ollama_available
    ollama_available=$(check_ollama_running)
    
    if [[ "$ollama_available" != "true" ]]; then
        log_error_with_context "Ollama is not running"
        log_info "Start Ollama with: ollama serve"
        log_info "Then install a model with: ollama pull llama3.2:3b"
        log_operation_end "$operation_name" 1 0
        return 1
    fi

    local exit_code=0
    if run_python_script_with_retry "--summarize" 1; then
        exit_code=0
    else
        exit_code=$?
        log_info "Check Ollama server status and model availability"
    fi
    
    local end_time
    end_time=$(date +%s)
    local duration
    duration=$(get_elapsed_time "$start_time" "$end_time")
    
    log_operation_end "$operation_name" "$exit_code" "$duration"
    return $exit_code
}

# Function: run_clear_library
# Purpose: Clear library entirely (removes all entries, PDFs, summaries, BibTeX, progress files)
# Args: None
# Returns:
#   0: Success
#   1: Failure or cancelled
# Side effects:
#   - Removes all library entries
#   - Deletes all PDFs, summaries, BibTeX file, progress file
#   - Requires confirmation
# Dependencies:
#   - check_python_script()
# Example:
#   run_clear_library
run_clear_library() {
    log_header_to_file "CLEAR LIBRARY ENTIRELY"
    log_header "CLEAR LIBRARY ENTIRELY"

    if ! cd "$REPO_ROOT" 2>/dev/null; then
        log_error "Cannot change to repository root: $REPO_ROOT"
        return 1
    fi

    echo
    log_warning "⚠️  WARNING: This will permanently delete:"
    log_warning "   • All library entries (data/library.json)"
    log_warning "   • All PDFs (data/pdfs/)"
    log_warning "   • All summaries (data/summaries/)"
    log_warning "   • Bibliography file (data/references.bib)"
    log_warning "   • Progress file (data/summarization_progress.json)"
    echo
    log_warning "This operation cannot be undone!"
    echo
    
    read -p "Type 'CLEAR' to confirm: " confirmation
    if [[ "$confirmation" != "CLEAR" ]]; then
        log_info "Operation cancelled"
        return 1
    fi
    
    log_info_to_file "Clearing library entirely..."
    log_info "Clearing library entirely..."
    
    python3 << 'PYTHON_EOF'
import sys
from pathlib import Path

try:
    from infrastructure.literature.library.clear import clear_library
    
    result = clear_library(confirm=False, interactive=False)
    
    if result.get("success"):
        print(f"✓ Library cleared successfully")
        print(f"  • Entries removed: {result.get('entries_removed', 0)}")
        print(f"  • PDFs removed: {result.get('pdfs_removed', 0)} ({result.get('pdfs_size_mb', 0):.1f} MB)")
        print(f"  • Summaries removed: {result.get('summaries_removed', 0)} ({result.get('summaries_size_mb', 0):.1f} MB)")
        sys.exit(0)
    else:
        print(f"✗ Library clear failed: {result.get('message', 'Unknown error')}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error clearing library: {e}")
    sys.exit(1)
PYTHON_EOF

    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log_success_to_file "Library cleared successfully"
        log_success "Library cleared successfully"
        return 0
    else
        log_error_to_file "Library clear failed"
        log_error "Library clear failed"
        return 1
    fi
}

# Function: run_literature_llm_operations
# Purpose: Execute advanced LLM operations (literature review, comparisons, etc.)
# Args: None
# Returns:
#   0: At least one operation succeeded
#   1: All operations failed or no operations selected
# Side effects:
#   - Prompts user to configure which operations to run (default: all)
#   - Executes selected operations with timeout protection
#   - Requires Ollama server running
# Dependencies:
#   - check_python_script()
#   - check_ollama_running()
#   - run_python_script_with_retry()
#   - with_timeout()
#   - prompt_yes_no_default()
# Example:
#   run_literature_llm_operations
run_literature_llm_operations() {
    log_header_to_file "ADVANCED LLM OPERATIONS (LITERATURE REVIEW, ETC.)"
    log_header "ADVANCED LLM OPERATIONS (LITERATURE REVIEW, ETC.)"

    if ! cd "$REPO_ROOT" 2>/dev/null; then
        log_error "Cannot change to repository root: $REPO_ROOT"
        return 1
    fi

    if ! check_python_script; then
        return 1
    fi

    # Clear LLM context first to start fresh
    echo
    log_info "Preparing LLM environment..."
    log_debug "Step 1: Clearing LLM context..."
    log_info "Clearing LLM context..."
    
    local context_status
    local context_stderr
    local context_stderr_file
    context_stderr_file=$(mktemp /tmp/llm_context_stderr.XXXXXX 2>/dev/null || echo "/tmp/llm_context_stderr.$$")
    
    context_status=$(python3 << 'PYTHON_EOF' 2> "$context_stderr_file"
import sys
import json
import traceback
try:
    from infrastructure.llm import LLMClient, LLMConfig
    
    config = LLMConfig.from_env()
    client = LLMClient(config=config)
    
    messages_before = len(client.context.messages)
    tokens_before = client.context.estimated_tokens
    
    client.reset()
    
    messages_after = len(client.context.messages)
    tokens_after = client.context.estimated_tokens
    
    status = {
        "success": True,
        "messages_cleared": messages_before,
        "tokens_cleared": tokens_before,
        "messages_after": messages_after,
        "tokens_after": tokens_after
    }
    print(json.dumps(status))
    sys.exit(0)
except ImportError as e:
    status = {
        "success": False,
        "error": f"Import error: {str(e)}",
        "error_type": "import_error",
        "traceback": traceback.format_exc(),
        "messages_cleared": 0,
        "tokens_cleared": 0
    }
    print(json.dumps(status))
    sys.exit(1)
except Exception as e:
    status = {
        "success": False,
        "error": str(e),
        "error_type": "unknown_error",
        "traceback": traceback.format_exc(),
        "messages_cleared": 0,
        "tokens_cleared": 0
    }
    print(json.dumps(status))
    sys.exit(1)
PYTHON_EOF
)
    
    local context_exit_code=$?
    context_stderr=$(cat "$context_stderr_file" 2>/dev/null || echo "")
    rm -f "$context_stderr_file"
    
    log_debug "Context clearing exit code: $context_exit_code"
    
    if [[ $context_exit_code -eq 0 ]]; then
        # Try to parse JSON response
        local json_output
        json_output=$(echo "$context_status" | grep -E '^\{.*\}$' | tail -1 || echo "$context_status")
        
        local success_flag
        success_flag=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('success', False))" 2>/dev/null || echo "false")
        
        if [[ "$success_flag" == "True" ]]; then
            local messages_cleared
            messages_cleared=$(echo "$json_output" | python3 -c "import sys, json; print(json.load(sys.stdin).get('messages_cleared', 0))" 2>/dev/null || echo "0")
            local tokens_cleared
            tokens_cleared=$(echo "$json_output" | python3 -c "import sys, json; print(json.load(sys.stdin).get('tokens_cleared', 0))" 2>/dev/null || echo "0")
            
            if [[ "$messages_cleared" -gt 0 ]] || [[ "$tokens_cleared" -gt 0 ]]; then
                log_success "Cleared $messages_cleared message(s) and ~$tokens_cleared token(s) from context"
            else
                log_info "Context was already empty"
            fi
        else
            local error_msg
            error_msg=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('error', 'Unknown error'))" 2>/dev/null || echo "Unknown error")
            log_warning "Failed to clear LLM context: $error_msg"
            log_debug "Context clearing error details: $json_output"
            if [[ -n "$context_stderr" ]]; then
                log_debug "Python stderr: ${context_stderr:0:200}"
            fi
        fi
    else
        # Python script failed - try to extract error from output
        local error_msg=""
        if [[ -n "$context_status" ]]; then
            local json_output
            json_output=$(echo "$context_status" | grep -E '^\{.*\}$' | tail -1 || echo "")
            if [[ -n "$json_output" ]]; then
                error_msg=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('error', 'Unknown error'))" 2>/dev/null || echo "Unknown error")
            fi
        fi
        
        if [[ -z "$error_msg" ]]; then
            error_msg="Python script exited with code $context_exit_code"
            if [[ -n "$context_stderr" ]]; then
                error_msg="${error_msg}. Error: ${context_stderr:0:200}"
            fi
        fi
        
        log_warning "Failed to clear LLM context: $error_msg"
        log_debug "Context clearing failed - exit code: $context_exit_code, output: ${context_status:0:200}"
    fi
    
    # Context clearing is non-fatal - continue anyway
    log_debug "Continuing after context clearing (non-fatal step)"

    # Ensure Ollama is ready (with restart if needed)
    echo
    log_debug "Step 2: Verifying Ollama server and model availability..."
    log_info "Verifying Ollama server and model availability..."
    
    if ! ensure_ollama_ready_with_restart; then
        log_error "Ollama is not ready. Cannot proceed with LLM operations."
        echo
        log_info "Troubleshooting steps:"
        log_info "  1. Check if Ollama is installed: ollama --version"
        log_info "  2. Start Ollama server: ollama serve"
        log_info "  3. Install a model: ollama pull <model-name>"
        log_info "  4. Check Ollama status: ollama ps"
        echo
        log_error "Please fix Ollama issues and try again."
        return 1
    fi
    
    log_debug "Ollama verification passed"

    # Select and verify model
    echo
    log_debug "Step 3: Selecting best available model..."
    log_info "Selecting best available model..."
    
    local model_info
    local model_stderr
    local model_stderr_file
    model_stderr_file=$(mktemp /tmp/llm_model_stderr.XXXXXX 2>/dev/null || echo "/tmp/llm_model_stderr.$$")
    
    model_info=$(python3 << 'PYTHON_EOF' 2> "$model_stderr_file"
import sys
import json
import traceback
try:
    from infrastructure.llm.utils.ollama import select_best_model, get_model_info
    from infrastructure.llm import LLMClient, LLMConfig
    
    base_url = "http://localhost:11434"
    model = select_best_model(base_url=base_url)
    
    if not model:
        status = {
            "success": False,
            "error": "No models available",
            "error_type": "no_models"
        }
        print(json.dumps(status))
        sys.exit(1)
    
    # Get model info
    info = get_model_info(model, base_url=base_url)
    model_size = info.get("size", 0) if info else 0
    model_size_gb = model_size / (1024**3) if model_size > 0 else 0
    
    # Test model with a simple query
    config = LLMConfig.from_env()
    config.default_model = model
    client = LLMClient(config)
    
    test_success, test_error = client.check_connection_detailed(timeout=5.0)
    
    status = {
        "success": True,
        "model": model,
        "model_size_gb": round(model_size_gb, 2),
        "test_success": test_success,
        "test_error": test_error if not test_success else None
    }
    print(json.dumps(status))
    sys.exit(0)
except ImportError as e:
    status = {
        "success": False,
        "error": f"Import error: {str(e)}",
        "error_type": "import_error",
        "traceback": traceback.format_exc()
    }
    print(json.dumps(status))
    sys.exit(1)
except Exception as e:
    status = {
        "success": False,
        "error": str(e),
        "error_type": "unknown_error",
        "traceback": traceback.format_exc()
    }
    print(json.dumps(status))
    sys.exit(1)
PYTHON_EOF
)
    
    local model_exit_code=$?
    model_stderr=$(cat "$model_stderr_file" 2>/dev/null || echo "")
    rm -f "$model_stderr_file"
    
    log_debug "Model selection exit code: $model_exit_code"
    
    if [[ $model_exit_code -ne 0 ]]; then
        # Try to extract error from JSON output
        local json_output
        json_output=$(echo "$model_info" | grep -E '^\{.*\}$' | tail -1 || echo "$model_info")
        
        local model_error=""
        local error_type=""
        if [[ -n "$json_output" ]]; then
            model_error=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('error', 'Unknown error'))" 2>/dev/null || echo "Unknown error")
            error_type=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('error_type', ''))" 2>/dev/null || echo "")
        fi
        
        if [[ -z "$model_error" ]] || [[ "$model_error" == "Unknown error" ]]; then
            model_error="Failed to select model (exit code: $model_exit_code)"
            if [[ -n "$model_stderr" ]]; then
                model_error="${model_error}. Error: ${model_stderr:0:200}"
            fi
        fi
        
        log_error "Model selection failed: $model_error"
        
        # Provide helpful error messages based on error type
        case "$error_type" in
            "no_models")
                log_info "No Ollama models are installed."
                log_info "Install a model with: ollama pull <model-name>"
                log_info "Example: ollama pull llama3:latest"
                ;;
            "import_error")
                log_info "Failed to import required Python modules."
                log_info "Check that infrastructure.llm modules are available."
                ;;
            *)
                log_info "Check Ollama server status: ollama ps"
                log_info "Verify models are installed: ollama list"
                ;;
        esac
        
        log_debug "Model selection error details: $json_output"
        if [[ -n "$model_stderr" ]]; then
            log_debug "Python stderr: ${model_stderr:0:200}"
        fi
        
        return 1
    fi
    
    log_debug "Model selection succeeded"
    
    # Parse model info from JSON
    local json_output
    json_output=$(echo "$model_info" | grep -E '^\{.*\}$' | tail -1 || echo "$model_info")
    
    local selected_model
    selected_model=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('model', 'unknown'))" 2>/dev/null || echo "unknown")
    local model_size_gb
    model_size_gb=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('model_size_gb', 0))" 2>/dev/null || echo "0")
    local test_success
    test_success=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('test_success', False))" 2>/dev/null || echo "false")
    
    if [[ "$selected_model" != "unknown" ]]; then
        log_success "Selected model: $selected_model"
        if [[ "$model_size_gb" != "0" ]]; then
            log_info "  • Model size: ${model_size_gb} GB"
        fi
        if [[ "$test_success" == "True" ]]; then
            log_info "  • Model connection test: OK"
        else
            log_warning "  • Model connection test: Warning (may still work)"
        fi
    else
        log_error "Failed to determine selected model from response"
        log_debug "Model info response: ${json_output:0:200}"
        return 1
    fi

    echo
    log_debug "Step 4: Displaying operations menu..."
    log_success "LLM environment ready. Proceeding to operations menu..."
    echo

    # Helper function to get operation info
    # Format: "display_name|operation_code|is_special"
    # is_special: "1" means call run_literature_summarize directly, "0" means use --llm-operation
    get_operation_info() {
        local op_num="$1"
        case "$op_num" in
            1) echo "Summarize papers|summarize|1" ;;
            2) echo "Literature review synthesis|review|0" ;;
            3) echo "Science communication narrative|communication|0" ;;
            4) echo "Comparative analysis|compare|0" ;;
            5) echo "Research gap identification|gaps|0" ;;
            6) echo "Citation network analysis|network|0" ;;
            *) echo "" ;;
        esac
    }

    echo
    log_info "Available LLM operations:"
    log_info "  1. Summarize papers (generate summaries for papers with PDFs)"
    log_info "  2. Literature review synthesis"
    log_info "  3. Science communication narrative"
    log_info "  4. Comparative analysis"
    log_info "  5. Research gap identification"
    log_info "  6. Citation network analysis"
    echo
    log_info "Configure which operations to run (default: all):"
    echo

    # Collect selected operations
    local selected_ops=()
    local op_num
    local op_info
    local op_display
    local op_code
    local op_special

    log_debug "Prompting user for operation selection..."
    for op_num in 1 2 3 4 5 6; do
        op_info=$(get_operation_info "$op_num")
        if [[ -z "$op_info" ]]; then
            log_debug "No operation info for option $op_num, skipping"
            continue
        fi
        
        op_display="${op_info%%|*}"
        op_code="${op_info#*|}"
        op_code="${op_code%|*}"
        op_special="${op_info##*|}"

        log_debug "Prompting for operation $op_num: $op_display"
        if prompt_yes_no_default "Run ${op_display}" "Y"; then
            selected_ops+=("${op_num}|${op_display}|${op_code}|${op_special}")
            log_debug "Operation $op_num ($op_display) selected"
        else
            log_debug "Operation $op_num ($op_display) not selected"
        fi
    done

    # Check if any operations were selected
    log_debug "Total operations selected: ${#selected_ops[@]}"
    if [[ ${#selected_ops[@]} -eq 0 ]]; then
        log_warning "No operations selected. Exiting."
        return 1
    fi

    echo
    log_info "Selected ${#selected_ops[@]} operation(s) to run"
    echo

    # Determine timeout (use LLM_SUMMARIZATION_TIMEOUT, fallback to LLM_TIMEOUT, then 600s)
    local timeout_seconds=600
    if [[ -n "${LLM_SUMMARIZATION_TIMEOUT:-}" ]] && validate_numeric "${LLM_SUMMARIZATION_TIMEOUT}" 1; then
        timeout_seconds="${LLM_SUMMARIZATION_TIMEOUT}"
    elif [[ -n "${LLM_TIMEOUT:-}" ]] && validate_numeric "${LLM_TIMEOUT}" 1; then
        timeout_seconds="${LLM_TIMEOUT}"
    fi

    log_debug "Using timeout: ${timeout_seconds}s for LLM operations"

    # Track results
    local total_ops=${#selected_ops[@]}
    local success_count=0
    local failure_count=0
    local failed_ops=()

    # Execute each selected operation
    local idx=0
    for op_info in "${selected_ops[@]}"; do
        ((idx++))
        # Parse operation info: format is "num|display|code|special"
        IFS='|' read -r op_num op_display op_code op_special <<< "$op_info"

        log_info "[${idx}/${total_ops}] Running: ${op_display}..."
        log_info_to_file "Starting LLM operation: ${op_display} (${op_code})"
        log_debug "Executing operation: $op_display (code: $op_code, special: $op_special)"

        local op_start_time
        op_start_time=$(date +%s)
        local op_exit_code=0
        local op_error_msg=""

        # Execute operation with timeout and error handling
        if [[ "$op_special" == "1" ]]; then
            # Special case: operation 1 (summarize) - call run_literature_summarize directly
            log_debug "Using special execution path for summarize operation"
            if with_timeout "$timeout_seconds" run_literature_summarize; then
                op_exit_code=0
                log_debug "Summarize operation completed successfully"
            else
                op_exit_code=$?
                if [[ $op_exit_code -eq 124 ]]; then
                    op_error_msg="timeout after ${timeout_seconds}s"
                else
                    op_error_msg="exit code ${op_exit_code}"
                fi
                log_debug "Summarize operation failed: $op_error_msg"
            fi
        else
            # Standard operations: use --llm-operation flag
            log_debug "Using standard execution path with --llm-operation flag: $op_code"
            log_debug "Command: python3 $PYTHON_SCRIPT --llm-operation $op_code"
            if with_timeout "$timeout_seconds" run_python_script_with_retry "--llm-operation ${op_code}" 1; then
                op_exit_code=0
                log_debug "LLM operation $op_code completed successfully"
            else
                op_exit_code=$?
                if [[ $op_exit_code -eq 124 ]]; then
                    op_error_msg="timeout after ${timeout_seconds}s"
                else
                    op_error_msg="exit code ${op_exit_code}"
                fi
                log_debug "LLM operation $op_code failed: $op_error_msg"
            fi
        fi

        local op_end_time
        op_end_time=$(date +%s)
        local op_duration
        op_duration=$(get_elapsed_time "$op_start_time" "$op_end_time")

        # Record result
        if [[ $op_exit_code -eq 0 ]]; then
            ((success_count++))
            log_success "${op_display} completed (duration: $(format_duration "$op_duration"))"
            log_success_to_file "${op_display} completed (duration: $(format_duration "$op_duration"))"
        else
            ((failure_count++))
            failed_ops+=("${op_display}")
            if [[ -n "$op_error_msg" ]]; then
                log_error "${op_display} failed: ${op_error_msg}"
                log_error_to_file "${op_display} failed: ${op_error_msg} (duration: $(format_duration "$op_duration"))"
            else
                log_error "${op_display} failed (duration: $(format_duration "$op_duration"))"
                log_error_to_file "${op_display} failed (exit code: ${op_exit_code}, duration: $(format_duration "$op_duration"))"
            fi
            log_info "Check Ollama server status and model availability"
        fi

        echo
    done

    # Display summary
    echo
    log_header "OPERATIONS SUMMARY"
    log_info "Operations completed: ${total_ops} attempted, ${success_count} succeeded, ${failure_count} failed"

    if [[ ${#failed_ops[@]} -gt 0 ]]; then
        log_warning "Failed operations:"
        for failed_op in "${failed_ops[@]}"; do
            log_warning "  • ${failed_op}"
        done
        echo
        log_info "To retry failed operations, run option 5.1 again and select only failed operations"
    fi

    echo

    # Return code: 0 if at least one succeeded, 1 if all failed
    if [[ $success_count -gt 0 ]]; then
        log_success_to_file "LLM operations completed: ${success_count}/${total_ops} succeeded"
        return 0
    else
        log_error_to_file "All LLM operations failed"
        return 1
    fi
}

