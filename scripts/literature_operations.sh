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
# Purpose: Execute meta-analysis pipeline: search → download → extract → meta-analysis
# Args: None
# Returns:
#   0: Success
#   1: Failure
# Side effects:
#   - Executes Python script with --meta-analysis flag
#   - Creates/updates bibliography and library files
#   - Downloads PDFs, extracts text, performs meta-analysis
# Dependencies:
#   - check_python_script()
#   - run_python_script_with_retry()
# Example:
#   run_literature_meta_analysis
run_literature_meta_analysis() {
    log_header_to_file "ORCHESTRATED META-ANALYSIS PIPELINE"
    log_header "ORCHESTRATED META-ANALYSIS PIPELINE"
    
    if ! cd "$REPO_ROOT" 2>/dev/null; then
        log_error "Cannot change to repository root: $REPO_ROOT"
        return 1
    fi
    
    if ! check_python_script; then
        return 1
    fi
    
    echo
    log_info_to_file "🔄 Starting meta-analysis pipeline..."
    log_info "🔄 Starting meta-analysis pipeline..."
    echo
    log_info "Pipeline stages:"
    log_info "  1️⃣  Search academic databases for keywords"
    log_info "  2️⃣  Download PDFs from available sources"
    log_info "  3️⃣  Extract text from PDFs (save to extracted_text/)"
    log_info "  4️⃣  Perform meta-analysis (PCA, keywords, authors, visualizations)"
    echo
    log_info "You will be prompted for:"
    log_info "  • Search keywords (comma-separated)"
    log_info "  • Number of results per keyword (default: 25)"
    log_info "  • Clear options (PDFs/Library - default: No)"
    echo
    
    local start_time=$(date +%s)
    log_debug "Pipeline start time: $start_time"
    
    # Use --meta-analysis which runs the meta-analysis pipeline interactively
    # Clear options are handled interactively in the Python script
    if run_python_script_with_retry "--meta-analysis" 1; then
        local end_time=$(date +%s)
        local duration=$(get_elapsed_time "$start_time" "$end_time")
        echo
        log_success_to_file "✅ Meta-analysis pipeline complete in $(format_duration "$duration")"
        log_success "✅ Meta-analysis pipeline complete in $(format_duration "$duration")"
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
        log_error_to_file "❌ Meta-analysis pipeline failed"
        log_error "❌ Meta-analysis pipeline failed"
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
#   0: Success
#   1: Failure (invalid choice or execution error)
# Side effects:
#   - Prompts user for operation choice
#   - Executes Python script with --llm-operation flag
#   - Requires Ollama server running
# Dependencies:
#   - check_python_script()
#   - check_ollama_running()
#   - run_python_script_with_retry()
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

    # Check Ollama availability
    log_info "Checking Ollama availability..."
    local ollama_available
    ollama_available=$(check_ollama_running)
    
    if [[ "$ollama_available" != "true" ]]; then
        log_error "Ollama is not running"
        log_info "Start Ollama with: ollama serve"
        return 1
    fi

    echo
    log_info "Available LLM operations:"
    log_info "  1. Summarize papers (generate summaries for papers with PDFs)"
    log_info "  2. Literature review synthesis"
    log_info "  3. Science communication narrative"
    log_info "  4. Comparative analysis"
    log_info "  5. Research gap identification"
    log_info "  6. Citation network analysis"
    echo

    read -p "Choose operation (1-6): " op_choice

    # Validate input
    if [[ ! "$op_choice" =~ ^[1-6]$ ]]; then
        log_error "Invalid choice: $op_choice (must be 1-6)"
        return 1
    fi

    local operation=""
    case $op_choice in
        1)
            # Run summarize directly
            run_literature_summarize
            return $?
            ;;
        2)
            operation="review"
            ;;
        3)
            operation="communication"
            ;;
        4)
            operation="compare"
            ;;
        5)
            operation="gaps"
            ;;
        6)
            operation="network"
            ;;
        *)
            log_error "Invalid choice: $op_choice"
            return 1
            ;;
    esac

    log_info_to_file "Running LLM operation: $operation (requires Ollama)..."
    log_info "Running LLM operation: $operation (requires Ollama)..."

    if run_python_script_with_retry "--llm-operation $operation" 1; then
        log_success_to_file "LLM operation complete"
        log_success "LLM operation complete"
        return 0
    else
        log_error_to_file "LLM operation failed"
        log_error "LLM operation failed"
        log_info "Check Ollama server status and model availability"
        return 1
    fi
}

