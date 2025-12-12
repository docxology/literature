#!/usr/bin/env bash

################################################################################
# Literature Operations Orchestrator
#
# Entry point for literature search and management operations with interactive
# menu and non-interactive command-line modes.
#
# ARCHITECTURE OVERVIEW
# =====================
#
# This script orchestrates literature processing workflows by delegating to
# Python infrastructure modules. It provides:
# - Interactive menu-driven interface
# - Non-interactive CLI mode with flags
# - Pipeline orchestration (search → download → extract → summarize/analyze)
# - Individual operation execution
# - Test suite execution with coverage analysis
#
# OPERATION FLOW
# ==============
#
# Interactive Mode:
#   1. Display menu with current library status
#   2. User selects operation (0-9)
#   3. Execute selected operation via Python script
#   4. Display results and return to menu
#
# Non-Interactive Mode:
#   1. Parse command-line arguments
#   2. Execute corresponding operation directly
#   3. Exit with appropriate code
#
# Orchestrated Pipelines:
#   0. Full Pipeline: search → download → extract → summarize
#   1. Meta-Analysis Pipeline: search → download → extract → meta-analysis
#
# Individual Operations (via 07_literature_search.py):
#   2. Search Only (network only - add to bibliography)
#   3. Download Only (network only - download PDFs)
#   4. Extract Text (local only - extract text from PDFs)
#   5. Summarize (requires Ollama - generate summaries)
#   6. Cleanup (local files only - remove papers without PDFs and orphaned files)
#   7. Advanced LLM Operations (requires Ollama)
#   9. Run Test Suite (with coverage analysis)
#   8. Exit
#
# EXIT CODES
# ==========
#   0 = Success
#   1 = Failure (error occurred)
#   2 = Skipped (optional stage skipped)
#
# ENVIRONMENT VARIABLES
# =====================
#   LOG_LEVEL          - Logging verbosity (0=DEBUG, 1=INFO, 2=WARN, 3=ERROR)
#   PIPELINE_LOG_FILE  - Optional log file path for file logging
#   PYTHONPATH         - Python module search path (auto-configured)
#
# TROUBLESHOOTING
# ===============
# - Python script not found: Ensure scripts/07_literature_search.py exists
# - Ollama not running: Start with 'ollama serve' for summarization/LLM ops
# - Permission errors: Check write permissions for data/ directory
# - Network errors: Verify internet connection for search/download operations
#
# EXAMPLES
# ========
#   # Interactive mode
#   ./run_literature.sh
#
#   # Non-interactive: search only
#   ./run_literature.sh --search
#
#   # Non-interactive: full pipeline
#   ./run_literature.sh --option 0
#
#   # With debug logging
#   LOG_LEVEL=0 ./run_literature.sh --search
#
#   # With file logging
#   PIPELINE_LOG_FILE=literature.log ./run_literature.sh
################################################################################

set -euo pipefail

# Source shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
source "$SCRIPT_DIR/scripts/bash_utils.sh"

# ============================================================================
# Constants
# ============================================================================

readonly COVERAGE_THRESHOLD=60
readonly PYTHON_SCRIPT="scripts/07_literature_search.py"
readonly MAX_TEST_OUTPUT_LINES=20
readonly MAX_FAILED_TEST_LINES=20

# ============================================================================
# Helper Functions
# ============================================================================

# ============================================================================
# Method Enhancement Utilities
# ============================================================================

# Function: with_timeout
# Purpose: Execute a command with timeout
# Args:
#   $1: Timeout in seconds
#   $2: Command to execute (as separate arguments)
# Returns:
#   0: Command completed successfully
#   124: Command timed out (timeout command exit code)
#   Other: Command exit code
# Side effects: Logs timeout if occurs
with_timeout() {
    local timeout_seconds="${1:-0}"
    shift
    
    if [[ $# -eq 0 ]]; then
        log_error_with_context "with_timeout: No command provided"
        return 1
    fi
    
    if ! validate_numeric "$timeout_seconds" 1; then
        log_error_with_context "with_timeout: Invalid timeout value: $timeout_seconds"
        return 1
    fi
    
    log_debug "Executing with timeout ${timeout_seconds}s: $*"
    
    # Check if timeout command is available
    if command -v timeout &> /dev/null; then
        # Use GNU timeout or BSD gtimeout
        if timeout "$timeout_seconds" "$@"; then
            return 0
        else
            local exit_code=$?
            if [[ $exit_code -eq 124 ]]; then
                log_error_with_context "Command timed out after ${timeout_seconds}s: $*" "$exit_code"
            fi
            return $exit_code
        fi
    elif command -v gtimeout &> /dev/null; then
        # macOS with coreutils
        if gtimeout "$timeout_seconds" "$@"; then
            return 0
        else
            local exit_code=$?
            if [[ $exit_code -eq 124 ]]; then
                log_error_with_context "Command timed out after ${timeout_seconds}s: $*" "$exit_code"
            fi
            return $exit_code
        fi
    else
        # Fallback: execute without timeout (log warning)
        log_warning "timeout command not available, executing without timeout: $*"
        "$@"
        return $?
    fi
}

# Function: setup_signal_handlers
# Purpose: Setup signal handlers for graceful shutdown
# Args: None
# Side effects: Sets up SIGINT and SIGTERM handlers
setup_signal_handlers() {
    local operation_name="${1:-operation}"
    
    # Trap SIGINT (Ctrl+C) and SIGTERM
    trap "log_warning '${operation_name} interrupted by user'; exit 130" INT TERM
    
    log_debug "Signal handlers set up for ${operation_name}"
}

# Function: cleanup_signal_handlers
# Purpose: Remove signal handlers
# Args: None
cleanup_signal_handlers() {
    trap - INT TERM
    log_debug "Signal handlers removed"
}

# Function: validate_menu_choice
# Purpose: Validate that a menu choice is numeric and within valid range (0-9)
# Args:
#   $1: Choice string to validate
#   $2: Maximum valid choice (optional, default: 9)
# Returns:
#   0: Valid choice
#   1: Invalid choice
validate_menu_choice() {
    local choice="${1:-}"
    local max_choice="${2:-9}"
    
    if [[ -z "$choice" ]]; then
        return 1
    fi
    
    if ! validate_numeric "$choice" 0 "$max_choice"; then
        return 1
    fi
    
    return 0
}

# Function: check_environment
# Purpose: Validate environment setup (Python, scripts, directories)
# Args: None
# Returns:
#   0: Environment is valid
#   1: Environment validation failed
# Side effects: Logs validation results
check_environment() {
    local errors=0
    
    log_debug "Validating environment..."
    
    # Check Python availability
    if ! command -v python3 &> /dev/null; then
        log_error_with_context "python3 not found in PATH"
        ((errors++))
    else
        local python_version
        python_version=$(python3 --version 2>&1 || echo "unknown")
        log_debug "Python version: $python_version"
    fi
    
    # Check repository root
    if [[ ! -d "$REPO_ROOT" ]]; then
        log_error_with_context "Repository root not found: $REPO_ROOT"
        ((errors++))
    fi
    
    # Check Python script
    if [[ ! -f "$REPO_ROOT/$PYTHON_SCRIPT" ]]; then
        log_error_with_context "Python script not found: $PYTHON_SCRIPT"
        log_info "Expected location: $REPO_ROOT/$PYTHON_SCRIPT"
        ((errors++))
    fi
    
    # Check data directory
    if [[ ! -d "$REPO_ROOT/data" ]]; then
        log_warning "Data directory not found: $REPO_ROOT/data (will be created if needed)"
    fi
    
    # Check write permissions for data directory
    if [[ -d "$REPO_ROOT/data" ]] && [[ ! -w "$REPO_ROOT/data" ]]; then
        log_error_with_context "Data directory is not writable: $REPO_ROOT/data"
        ((errors++))
    fi
    
    if [[ $errors -eq 0 ]]; then
        log_debug "Environment validation passed"
        return 0
    else
        log_error_with_context "Environment validation failed with $errors error(s)"
        return 1
    fi
}

# Function: check_python_script
# Purpose: Verify that the Python orchestrator script exists
# Args: None
# Returns:
#   0: Script exists
#   1: Script not found
# Side effects:
#   - Logs error if script not found
check_python_script() {
    if [[ ! -f "$REPO_ROOT/$PYTHON_SCRIPT" ]]; then
        log_error_with_context "Python script not found: $PYTHON_SCRIPT"
        log_info "Expected location: $REPO_ROOT/$PYTHON_SCRIPT"
        return 1
    fi
    return 0
}

# Function: run_python_script_with_retry
# Purpose: Execute Python script with optional retry logic for transient failures
# Args:
#   $1: Command arguments to pass to Python script (space-separated)
#   $2: Maximum retry attempts (default: 1, no retry)
# Returns:
#   0: Success
#   1: Failure after retries
# Side effects:
#   - Executes Python script, may retry on failure
#   - Logs attempts and failures
run_python_script_with_retry() {
    local args="$1"
    local max_retries="${2:-1}"
    local attempt=1
    local exit_code=0
    
    log_debug "Executing: python3 $PYTHON_SCRIPT $args"
    
    while [[ $attempt -le $max_retries ]]; do
        if [[ $attempt -gt 1 ]]; then
            log_warning "Retry attempt $attempt/$max_retries for: $PYTHON_SCRIPT $args"
        fi
        
        if python3 "$REPO_ROOT/$PYTHON_SCRIPT" $args; then
            return 0
        else
            exit_code=$?
            if [[ $attempt -lt $max_retries ]]; then
                log_warning "Command failed (exit code: $exit_code), retrying..."
                sleep 1
            fi
        fi
        ((attempt++))
    done
    
    log_error "Command failed after $max_retries attempt(s): $PYTHON_SCRIPT $args"
    return $exit_code
}

# ============================================================================
# Error Handling Utilities
# ============================================================================

# Function: _get_error_context
# Purpose: Get error context for logging (function name, line number)
# Returns: Context string
_get_error_context() {
    local func_name="${FUNCNAME[2]:-unknown}"
    local line_num="${BASH_LINENO[1]:-0}"
    local script_name="${BASH_SOURCE[1]:-unknown}"
    script_name=$(basename "$script_name")
    
    if [[ "${LOG_LEVEL:-1}" == "0" ]] || [[ "${LOG_STRUCTURED:-false}" == "true" ]]; then
        echo "[${script_name}:${func_name}:${line_num}]"
    fi
}

# Function: log_error_with_context
# Purpose: Log error with context information
# Args:
#   $1: Error message
#   $2: Exit code (optional)
log_error_with_context() {
    local message="${1:-}"
    local exit_code="${2:-}"
    
    if [[ -z "$message" ]]; then
        return 0
    fi
    
    local context
    context=$(_get_error_context)
    local full_message="$message"
    
    if [[ -n "$context" ]]; then
        full_message="${context} ${message}"
    fi
    
    if [[ -n "$exit_code" ]] && [[ "$exit_code" =~ ^[0-9]+$ ]]; then
        full_message="${full_message} (exit code: ${exit_code})"
    fi
    
    log_error "$full_message"
    log_error_to_file "$full_message"
}

# Function: safe_execute
# Purpose: Safely execute a command with error handling
# Args:
#   $@: Command and arguments to execute
# Returns: Exit code of the command
# Side effects: Logs errors with context
safe_execute() {
    local exit_code=0
    local cmd_args=("$@")
    
    if [[ ${#cmd_args[@]} -eq 0 ]]; then
        log_error_with_context "safe_execute: No command provided"
        return 1
    fi
    
    log_debug "Executing: ${cmd_args[*]}"
    
    # Execute command
    "${cmd_args[@]}" || exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        log_error_with_context "Command failed: ${cmd_args[*]}" "$exit_code"
    fi
    
    return $exit_code
}

# Function: capture_command_output
# Purpose: Capture stdout and stderr from a command for analysis (safe version)
# Args:
#   $1: Command to execute (as array elements, not string)
#   $2: Output file path for stdout
#   $3: Error file path for stderr (optional)
# Returns:
#   Exit code of the command
# Side effects:
#   - Creates output files with command output
# Note: This function expects command as separate arguments, not a string
# Example: capture_command_output python3 script.py arg1 arg2 output.txt error.txt
capture_command_output() {
    local output_file="${1:-}"
    local error_file="${2:-}"
    shift 2 2>/dev/null || shift 1 2>/dev/null || true
    
    # Validate inputs
    if [[ -z "$output_file" ]]; then
        log_error_with_context "capture_command_output: Missing output file"
        return 1
    fi
    
    if [[ $# -eq 0 ]]; then
        log_error_with_context "capture_command_output: No command provided"
        return 1
    fi
    
    # Ensure output directory exists
    local output_dir
    output_dir=$(dirname "$output_file")
    if [[ -n "$output_dir" ]] && [[ ! -d "$output_dir" ]]; then
        mkdir -p "$output_dir" 2>/dev/null || {
            log_error_with_context "capture_command_output: Cannot create output directory: $output_dir"
            return 1
        }
    fi
    
    # Execute command and capture output
    local exit_code=0
    if [[ -n "$error_file" ]]; then
        # Ensure error file directory exists
        local error_dir
        error_dir=$(dirname "$error_file")
        if [[ -n "$error_dir" ]] && [[ ! -d "$error_dir" ]]; then
            mkdir -p "$error_dir" 2>/dev/null || {
                log_error_with_context "capture_command_output: Cannot create error directory: $error_dir"
                return 1
            }
        fi
        "$@" > "$output_file" 2> "$error_file" || exit_code=$?
    else
        "$@" > "$output_file" 2>&1 || exit_code=$?
    fi
    
    if [[ $exit_code -ne 0 ]]; then
        log_debug "Command exited with code $exit_code: $*"
    fi
    
    return $exit_code
}

# ============================================================================
# Menu Display Functions
# ============================================================================

# Function: get_library_stats_display
# Purpose: Retrieve and display library statistics using Python helper
# Args: None
# Returns:
#   0: Success (or graceful failure)
# Side effects:
#   - Executes Python code to fetch library stats
#   - Outputs statistics to stdout
# Dependencies:
#   - infrastructure.literature.library.stats module
# Example:
#   get_library_stats_display
get_library_stats_display() {
    log_debug "Fetching library statistics"
    
    if ! cd "$REPO_ROOT" 2>/dev/null; then
        log_error "Cannot change to repository root: $REPO_ROOT"
        echo "  • Library: Not available"
        return 1
    fi
    
    python3 << 'PYTHON_EOF'
import sys
from pathlib import Path

try:
    from infrastructure.literature.library.stats import get_library_statistics, format_library_stats_display
    stats = get_library_statistics()
    print(format_library_stats_display(stats))
except Exception as e:
    print("  • Library: Not available")
PYTHON_EOF
}

# Function: display_menu
# Purpose: Render the interactive menu with library status and options
# Args: None
# Returns: None
# Side effects:
#   - Clears screen
#   - Displays menu to stdout
#   - Calls get_library_stats_display()
# Dependencies:
#   - get_library_stats_display()
#   - Color codes from bash_utils.sh
# Example:
#   display_menu
display_menu() {
    clear
    
    # Menu header
    log_info_to_file "============================================================"
    log_info_to_file "  Literature Operations Menu"
    log_info_to_file "============================================================"
    echo -e "${BOLD}${BLUE}"
    echo "============================================================"
    echo "  Literature Operations Menu"
    echo "============================================================"
    echo -e "${NC}"
    echo
    
    # Display library statistics
    echo -e "${BOLD}${CYAN}Current Library Status:${NC}"
    get_library_stats_display
    echo
    
    # Menu options
    echo -e "${BOLD}Orchestrated Pipelines:${NC}"
    echo -e "  0. ${GREEN}Full Pipeline${NC} ${YELLOW}(search → download → extract → summarize)${NC}"
    echo -e "  1. ${GREEN}Meta-Analysis Pipeline${NC} ${YELLOW}(search → download → extract → meta-analysis)${NC}"
    echo
    echo -e "${BOLD}Individual Operations (via 07_literature_search.py):${NC}"
    echo -e "  2. Search Only ${CYAN}(network only - add to bibliography)${NC}"
    echo -e "  3. Download Only ${CYAN}(network only - download PDFs)${NC}"
    echo -e "  4. Extract Text ${CYAN}(local only - extract text from PDFs)${NC}"
    echo -e "  5. Summarize ${YELLOW}(requires Ollama - generate summaries)${NC}"
    echo -e "  6. Cleanup ${CYAN}(local files only - remove papers without PDFs and orphaned files)${NC}"
    echo -e "  7. Advanced LLM Operations ${YELLOW}(requires Ollama)${NC}"
    echo
    echo -e "${BOLD}Testing:${NC}"
    echo -e "  9. Run Test Suite ${CYAN}(with coverage analysis)${NC}"
    echo
    echo "  8. Exit"
    echo
    
    # Footer with environment info
    echo -e "${BLUE}============================================================${NC}"
    echo -e "  Repository: ${CYAN}$REPO_ROOT${NC}"
    echo -e "  Python: ${CYAN}$(python3 --version 2>&1)${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo
}

# ============================================================================
# Literature Operation Functions
# ============================================================================

# Function: run_literature_search_all
# Purpose: Execute full orchestrated pipeline: search → download → extract → summarize
# Args: None
# Returns:
#   0: Success
#   1: Failure
# Side effects:
#   - Executes Python script with --search flag
#   - Creates/updates bibliography and library files
#   - Downloads PDFs, extracts text, generates summaries
# Dependencies:
#   - check_python_script()
#   - run_python_script_with_retry()
# Example:
#   run_literature_search_all
run_literature_search_all() {
    log_header_to_file "ORCHESTRATED LITERATURE PIPELINE"
    log_header "ORCHESTRATED LITERATURE PIPELINE"
    
    local operation_name="full_pipeline"
    local start_time
    start_time=$(date +%s)
    
    log_operation_start "$operation_name" "Full orchestrated pipeline (search → download → extract → summarize)"
    
    if ! cd "$REPO_ROOT" 2>/dev/null; then
        log_error_with_context "Cannot change to repository root: $REPO_ROOT"
        log_operation_end "$operation_name" 1 0
        return 1
    fi
    
    if ! check_environment; then
        log_operation_end "$operation_name" 1 0
        return 1
    fi
    
    if ! check_python_script; then
        log_operation_end "$operation_name" 1 0
        return 1
    fi
    
    echo
    log_info "Pipeline stages:"
    log_info "  1️⃣  Search academic databases for keywords"
    log_info "  2️⃣  Download PDFs from available sources"
    log_info "  3️⃣  Extract text from PDFs (save to extracted_text/)"
    log_info "  4️⃣  Generate AI-powered summaries (requires Ollama)"
    echo
    log_info "You will be prompted for:"
    log_info "  • Search keywords (comma-separated)"
    log_info "  • Number of results per keyword (default: 25)"
    log_info "  • Clear options (PDFs/Summaries/Library - default: No)"
    echo
    
    log_debug "Pipeline start time: $start_time"
    
    # Use --search which runs the full pipeline interactively
    # Clear options are handled interactively in the Python script
    local exit_code=0
    if run_python_script_with_retry "--search" 1; then
        exit_code=0
    else
        exit_code=$?
    fi
    
    local end_time
    end_time=$(date +%s)
    local duration
    duration=$(get_elapsed_time "$start_time" "$end_time")
    
    echo
    if [[ $exit_code -eq 0 ]]; then
        log_info "📁 Output locations:"
        log_info "  • Bibliography: data/references.bib"
        log_info "  • JSON index: data/library.json"
        log_info "  • PDFs: data/pdfs/"
        log_info "  • Summaries: data/summaries/"
        echo
        log_operation_end "$operation_name" 0 "$duration"
    else
        log_operation_end "$operation_name" "$exit_code" "$duration"
    fi
    
    return $exit_code
}

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

# Function: run_literature_cleanup
# Purpose: Clean up library by removing papers without PDFs and orphaned files
# Args: None
# Returns:
#   0: Success
#   1: Failure
# Side effects:
#   - Executes Python script with --cleanup flag
#   - Removes entries from library.json
#   - Deletes orphaned files from filesystem
#   - No network required
# Dependencies:
#   - check_python_script()
#   - run_python_script_with_retry()
# Example:
#   run_literature_cleanup
run_literature_cleanup() {
    log_header_to_file "CLEANUP LIBRARY (REMOVE PAPERS WITHOUT PDFs AND ORPHANED FILES)"
    log_header "CLEANUP LIBRARY (REMOVE PAPERS WITHOUT PDFs AND ORPHANED FILES)"

    if ! cd "$REPO_ROOT" 2>/dev/null; then
        log_error "Cannot change to repository root: $REPO_ROOT"
        return 1
    fi

    if ! check_python_script; then
        return 1
    fi

    log_info_to_file "Cleaning up library by removing papers without PDFs and deleting orphaned files (local files only)..."
    log_info "Cleaning up library by removing papers without PDFs and deleting orphaned files (local files only)..."

    if run_python_script_with_retry "--cleanup" 1; then
        log_success_to_file "Library cleanup complete"
        log_success "Library cleanup complete"
        return 0
    else
        log_error_to_file "Library cleanup failed"
        log_error "Library cleanup failed"
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
    log_info "  1. Literature review synthesis"
    log_info "  2. Science communication narrative"
    log_info "  3. Comparative analysis"
    log_info "  4. Research gap identification"
    log_info "  5. Citation network analysis"
    echo

    read -p "Choose operation (1-5): " op_choice

    # Validate input
    if [[ ! "$op_choice" =~ ^[1-5]$ ]]; then
        log_error "Invalid choice: $op_choice (must be 1-5)"
        return 1
    fi

    local operation=""
    case $op_choice in
        1)
            operation="review"
            ;;
        2)
            operation="communication"
            ;;
        3)
            operation="compare"
            ;;
        4)
            operation="gaps"
            ;;
        5)
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

# Function: check_ollama_running
# Purpose: Check if Ollama server is running using Python helper
# Args: None
# Returns:
#   0: Ollama is running (prints "true")
#   1: Ollama is not running or check failed (prints "false")
# Side effects:
#   - Executes Python code to check Ollama status
#   - Outputs "true" or "false" to stdout
# Dependencies:
#   - infrastructure.llm.utils.ollama module
# Example:
#   if [[ "$(check_ollama_running)" == "true" ]]; then
#       echo "Ollama is running"
#   fi
check_ollama_running() {
    log_debug "Checking Ollama server status"
    
    if ! cd "$REPO_ROOT" 2>/dev/null; then
        echo "false"
        return 1
    fi
    
    python3 << 'PYTHON_EOF'
import sys
try:
    from infrastructure.llm.utils.ollama import is_ollama_running
    if is_ollama_running():
        print("true")
        sys.exit(0)
    else:
        print("false")
        sys.exit(0)
except Exception:
    print("false")
    sys.exit(0)
PYTHON_EOF
}

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

# Function: parse_test_results
# Purpose: Parse pytest test results with improved regex patterns
# Args:
#   $1: Test output file path
# Returns: Sets global variables: total_tests, passed_tests, failed_tests, skipped_tests
parse_test_results() {
    local output_file="${1:-}"
    
    if [[ -z "$output_file" ]] || [[ ! -f "$output_file" ]]; then
        total_tests=0
        passed_tests=0
        failed_tests=0
        skipped_tests=0
        return 1
    fi
    
    # Improved regex patterns for pytest output
    # Matches formats like: "1518 passed, 19 skipped in 299.85s"
    # or: "================= 1518 passed, 19 skipped in 299.85s ================="
    local summary_line
    summary_line=$(grep -iE "(passed|failed|skipped|error)" "$output_file" | grep -E "[0-9]+\s+(passed|failed|skipped|error)" | tail -1 || echo "")
    
    if [[ -n "$summary_line" ]]; then
        # Extract numbers with improved patterns
        # Handle both "N passed" and "N passed," formats
        passed_tests=$(echo "$summary_line" | grep -oE "\b[0-9]+\s+passed\b" | grep -oE "[0-9]+" | head -1 || echo "0")
        failed_tests=$(echo "$summary_line" | grep -oE "\b[0-9]+\s+failed\b" | grep -oE "[0-9]+" | head -1 || echo "0")
        skipped_tests=$(echo "$summary_line" | grep -oE "\b[0-9]+\s+skipped\b" | grep -oE "[0-9]+" | head -1 || echo "0")
        local error_tests
        error_tests=$(echo "$summary_line" | grep -oE "\b[0-9]+\s+error\b" | grep -oE "[0-9]+" | head -1 || echo "0")
        
        # Add errors to failed count
        if validate_numeric "$error_tests" 0 && [[ "$error_tests" -gt 0 ]]; then
            failed_tests=$((failed_tests + error_tests))
        fi
        
        # Validate and calculate total
        if validate_numeric "$passed_tests" 0 && validate_numeric "$failed_tests" 0 && validate_numeric "$skipped_tests" 0; then
            total_tests=$((passed_tests + failed_tests + skipped_tests))
        else
            log_debug "Could not parse test counts reliably from: $summary_line"
            total_tests=0
            passed_tests=0
            failed_tests=0
            skipped_tests=0
            return 1
        fi
    else
        log_debug "No test summary line found in output"
        total_tests=0
        passed_tests=0
        failed_tests=0
        skipped_tests=0
        return 1
    fi
    
    return 0
}

# Function: parse_coverage_results
# Purpose: Parse pytest coverage results with improved patterns
# Args:
#   $1: Test output file path
# Returns: Sets global variables: total_coverage_full, total_coverage_int, coverage_summary
parse_coverage_results() {
    local output_file="${1:-}"
    
    if [[ -z "$output_file" ]] || [[ ! -f "$output_file" ]]; then
        total_coverage_full=""
        total_coverage_int=""
        coverage_summary=""
        return 1
    fi
    
    # Extract coverage summary table (more lines for better capture)
    coverage_summary=$(grep -A 30 "^TOTAL" "$output_file" | head -35 || echo "")
    
    # Improved pattern for coverage percentage
    # Matches formats like: "61.22%" or "TOTAL ... 61.22%"
    local total_line
    total_line=$(grep "^TOTAL" "$output_file" | head -1 || echo "")
    
    if [[ -n "$total_line" ]]; then
        # Extract percentage (last field or field containing %)
        total_coverage_full=$(echo "$total_line" | awk '{for(i=NF;i>=1;i--) if($i ~ /[0-9]+\.[0-9]+%/) {print $i; exit}}' || echo "")
        
        # If not found, try simpler pattern
        if [[ -z "$total_coverage_full" ]]; then
            total_coverage_full=$(echo "$total_line" | grep -oE "[0-9]+\.[0-9]+%" | head -1 || echo "")
        fi
        
        # Extract integer part
        if [[ -n "$total_coverage_full" ]]; then
            total_coverage_int=$(echo "$total_coverage_full" | grep -oE "^[0-9]+" | head -1 || echo "")
        fi
    fi
    
    if [[ -z "$total_coverage_full" ]]; then
        log_debug "Could not extract coverage percentage from output"
        return 1
    fi
    
    return 0
}

# Function: run_test_suite
# Purpose: Execute test suite with coverage analysis and detailed reporting
# Args: None
# Returns:
#   0: All tests passed
#   1: Tests failed or environment issues
# Side effects:
#   - Executes pytest with coverage
#   - Creates HTML coverage report in htmlcov/
#   - Parses and displays test results
#   - Creates temporary files for output capture
# Dependencies:
#   - pytest and pytest-cov installed
#   - check_ollama_running()
# Example:
#   run_test_suite
run_test_suite() {
    log_header_to_file "RUN TEST SUITE (WITH COVERAGE ANALYSIS)"
    log_header "RUN TEST SUITE (WITH COVERAGE ANALYSIS)"

    if ! cd "$REPO_ROOT" 2>/dev/null; then
        log_error "Cannot change to repository root: $REPO_ROOT"
        return 1
    fi

    echo
    log_info_to_file "Checking test environment..."
    log_info "Checking test environment..."

    # Check if pytest is installed
    if ! command -v pytest &> /dev/null; then
        log_error_to_file "pytest is not installed"
        log_error "pytest is not installed"
        log_info "Install with: pip install pytest pytest-cov"
        return 1
    fi

    # Check if pytest-cov is installed
    if ! python3 -c "import pytest_cov" 2>/dev/null; then
        log_error_to_file "pytest-cov is not installed"
        log_error "pytest-cov is not installed"
        log_info "Install with: pip install pytest-cov"
        return 1
    fi

    # Check Ollama availability
    log_info "Checking Ollama availability..."
    local ollama_available
    ollama_available=$(check_ollama_running)
    
    local pytest_args=()
    local test_mode=""
    
    if [[ "$ollama_available" == "true" ]]; then
        log_success_to_file "Ollama is running - will include all tests (including requires_ollama)"
        log_success "Ollama is running - will include all tests (including requires_ollama)"
        test_mode="full"
    else
        log_warning_to_file "Ollama is not running - will skip requires_ollama tests"
        log_warning "Ollama is not running - will skip requires_ollama tests"
        pytest_args+=("-m" "not requires_ollama")
        test_mode="partial"
    fi

    echo
    log_info_to_file "Running test suite with coverage analysis..."
    log_info "Test mode: $test_mode"
    log_info "Modules: core → llm → literature"
    echo

    local start_time=$(date +%s)
    local test_output_file
    test_output_file=$(mktemp)
    local coverage_output_file
    coverage_output_file=$(mktemp)
    
    log_debug "Test output file: $test_output_file"
    log_debug "Coverage output file: $coverage_output_file"

    # Run pytest with coverage
    # Use tee to stream output in real-time while capturing for parsing
    # Note: Excluding test_llm_review.py due to import errors (tests script that doesn't exist in this repo)
    local pytest_cmd=(
        pytest
        --cov=infrastructure
        --cov-report=term
        --cov-report=term-missing
        --cov-report=html
        --tb=short
        -v
        --durations=10
        --ignore=tests/infrastructure/llm/test_llm_review.py
    )
    
    # Add Ollama marker filter if needed
    if [[ ${#pytest_args[@]} -gt 0 ]]; then
        pytest_cmd+=("${pytest_args[@]}")
    fi
    
    pytest_cmd+=(tests/)
    
    log_debug "Pytest command: ${pytest_cmd[*]}"
    
    # Stream output in real-time using tee while capturing for parsing
    log_info_to_file "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info_to_file "TEST EXECUTION"
    log_info_to_file "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}TEST EXECUTION${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    
    local test_exit_code=0
    if "${pytest_cmd[@]}" 2>&1 | tee "$test_output_file"; then
        test_exit_code=0
    else
        test_exit_code=${PIPESTATUS[0]}
    fi

    local end_time=$(date +%s)
    local duration
    duration=$(get_elapsed_time "$start_time" "$end_time")
    
    log_debug "Test execution duration: $duration seconds"
    log_debug "Test exit code: $test_exit_code"

    # Parse test results using improved parsing function
    local total_tests passed_tests failed_tests skipped_tests
    if parse_test_results "$test_output_file"; then
        log_debug "Parsed test results: total=$total_tests, passed=$passed_tests, failed=$failed_tests, skipped=$skipped_tests"
    else
        log_debug "Failed to parse test results, using defaults"
        total_tests=0
        passed_tests=0
        failed_tests=0
        skipped_tests=0
    fi

    # Display concise summary only if we have parsed values
    if [[ "$total_tests" -gt 0 ]]; then
        echo
        log_info_to_file "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log_info_to_file "TEST SUMMARY"
        log_info_to_file "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BOLD}TEST SUMMARY${NC}"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo
        log_info_to_file "Total tests: $total_tests"
        if [[ "$passed_tests" -gt 0 ]]; then
            log_success_to_file "Passed: $passed_tests"
        fi
        if [[ "$failed_tests" -gt 0 ]]; then
            log_error_to_file "Failed: $failed_tests"
        fi
        if [[ "$skipped_tests" -gt 0 ]]; then
            log_warning_to_file "Skipped: $skipped_tests"
        fi
        log_info_to_file "Duration: $(format_duration "$duration")"
    fi

    # Parse coverage results using improved parsing function
    echo
    log_info_to_file "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info_to_file "COVERAGE ANALYSIS"
    log_info_to_file "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}COVERAGE ANALYSIS${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo

    local coverage_summary total_coverage_full total_coverage_int
    if parse_coverage_results "$test_output_file"; then
        log_debug "Parsed coverage results: ${total_coverage_full}"
        
        if [[ -n "$coverage_summary" ]]; then
            echo "$coverage_summary"
        else
            log_warning_to_file "Could not parse coverage summary from output"
        fi
        
        # Display coverage status
        if validate_numeric "$total_coverage_int" 0 100; then
            echo
            if [[ "$total_coverage_int" -ge $COVERAGE_THRESHOLD ]]; then
                log_success_to_file "Overall coverage: ${total_coverage_full} (meets ${COVERAGE_THRESHOLD}% threshold)"
            else
                log_warning_to_file "Overall coverage: ${total_coverage_full} (below ${COVERAGE_THRESHOLD}% threshold)"
            fi
        else
            log_debug "Invalid coverage percentage: ${total_coverage_int}"
            if [[ -z "$coverage_summary" ]]; then
                log_warning_to_file "Coverage data not available - check pytest output above"
            fi
        fi
    else
        log_warning_to_file "Could not parse coverage results from output"
    fi

    # Check for HTML coverage report
    if [[ -d "htmlcov" ]]; then
        echo
        log_info_to_file "📊 HTML coverage report generated: htmlcov/index.html"
        log_info "   Open with: open htmlcov/index.html"
    fi

    # List failed tests if any
    if [[ "$failed_tests" -gt 0 ]] || [[ "$test_exit_code" != "0" ]]; then
        echo
        log_info_to_file "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log_info_to_file "FAILED TESTS"
        log_info_to_file "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BOLD}FAILED TESTS${NC}"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo
        
        # Extract failed test names with limit
        local failed_test_lines
        failed_test_lines=$(grep -E "FAILED|ERROR" "$test_output_file" | head -$MAX_FAILED_TEST_LINES || echo "")
        if [[ -n "$failed_test_lines" ]]; then
            echo "$failed_test_lines"
        else
            log_info_to_file "Review test output above for failure details"
        fi
    fi

    # Cleanup temp files
    rm -f "$test_output_file" "$coverage_output_file"
    log_debug "Cleaned up temporary files"

    echo
    if [[ "$test_exit_code" == "0" ]]; then
        log_success_to_file "✅ Test suite completed successfully"
        return 0
    else
        log_error_to_file "❌ Test suite completed with failures (exit code: $test_exit_code)"
        return 1
    fi
}

# ============================================================================
# Menu Handler Functions
# ============================================================================

# Function: handle_menu_choice
# Purpose: Route menu choice to appropriate operation function
# Args:
#   $1: Menu choice (0-9)
# Returns:
#   0: Success or exit chosen
#   1: Invalid choice or operation failure
# Side effects:
#   - Executes corresponding operation function
#   - Tracks operation duration
# Dependencies:
#   - All run_literature_* functions
#   - run_test_suite()
# Example:
#   handle_menu_choice "0"
handle_menu_choice() {
    local choice="$1"
    local start_time end_time duration
    local exit_code=0
    
    # Validate choice
    if ! validate_menu_choice "$choice"; then
        log_error "Invalid menu choice format: $choice"
        log_info "Please enter a number between 0 and 9"
        return 1
    fi
    
    start_time=$(date +%s)
    log_debug "Handling menu choice: $choice"
    
    case "$choice" in
        0)
            run_literature_search_all
            exit_code=$?
            ;;
        1)
            run_literature_meta_analysis
            exit_code=$?
            ;;
        2)
            run_literature_search
            exit_code=$?
            ;;
        3)
            run_literature_download
            exit_code=$?
            ;;
        4)
            run_literature_extract_text
            exit_code=$?
            ;;
        5)
            run_literature_summarize
            exit_code=$?
            ;;
        6)
            run_literature_cleanup
            exit_code=$?
            ;;
        7)
            run_literature_llm_operations
            exit_code=$?
            ;;
        8)
            # Exit
            log_debug "Exit chosen"
            return 0
            ;;
        9)
            run_test_suite
            exit_code=$?
            ;;
        *)
            log_error "Invalid option: $choice"
            log_info "Please enter a number between 0 and 9"
            exit_code=1
            ;;
    esac
    
    end_time=$(date +%s)
    duration=$(get_elapsed_time "$start_time" "$end_time")
    
    echo
    log_info_to_file "Operation completed in $(format_duration "$duration")"
    log_info "Operation completed in $(format_duration "$duration")"
    return $exit_code
}

# Function: run_option_sequence
# Purpose: Execute a sequence of menu options in order, stopping on first failure
# Args:
#   $@: Array of menu option numbers
# Returns:
#   0: All operations succeeded
#   1: Operation failed (returns exit code of failed operation)
# Side effects:
#   - Executes operations in sequence
#   - Stops on first failure
# Dependencies:
#   - handle_menu_choice()
# Example:
#   run_option_sequence "0" "2" "3"
run_option_sequence() {
    local -a options=("$@")
    local exit_code=0

    if [[ ${#options[@]} -gt 0 ]]; then
        log_info_to_file "Running sequence: ${options[*]}"
        log_info "Running sequence: ${options[*]}"
    fi

    for opt in "${options[@]}"; do
        log_debug "Executing sequence option: $opt"
        handle_menu_choice "$opt"
        exit_code=$?
        if [[ $exit_code -ne 0 ]]; then
            log_error_to_file "Sequence aborted at option $opt (exit code $exit_code)"
            log_error "Sequence aborted at option $opt (exit code $exit_code)"
            return $exit_code
        fi
    done

    return $exit_code
}

# ============================================================================
# Non-Interactive Mode Functions
# ============================================================================

# Function: run_non_interactive
# Purpose: Execute operation in non-interactive mode (from command-line flags)
# Args:
#   $1: Option string (menu number or sequence)
# Returns:
#   Exits with operation exit code
# Side effects:
#   - Executes operation and exits (does not return to menu)
# Dependencies:
#   - parse_choice_sequence()
#   - run_option_sequence()
#   - handle_menu_choice()
# Example:
#   run_non_interactive "0"
run_non_interactive() {
    local option="$1"
    
    log_header_to_file "NON-INTERACTIVE MODE"
    log_header "NON-INTERACTIVE MODE"

    if parse_choice_sequence "$option" && [[ ${#SHORTHAND_CHOICES[@]} -gt 1 ]]; then
        log_info_to_file "Running shorthand sequence: ${SHORTHAND_CHOICES[*]}"
        log_info "Running shorthand sequence: ${SHORTHAND_CHOICES[*]}"
        run_option_sequence "${SHORTHAND_CHOICES[@]}"
        exit $?
    fi

    log_info_to_file "Running option: $option"
    log_info "Running option: $option"
    handle_menu_choice "$option"
    exit $?
}

# ============================================================================
# Help Functions
# ============================================================================

# Function: show_help
# Purpose: Display comprehensive help message with usage examples
# Args: None
# Returns: None
# Side effects:
#   - Outputs help text to stdout
# Example:
#   show_help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Literature Operations Orchestrator"
    echo
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --debug             Enable debug logging (equivalent to LOG_LEVEL=0)"
    echo
    echo "Literature Operations:"
    echo "  --search            Search literature (network only, add to bibliography)"
    echo "  --download          Download PDFs (network only, for bibliography entries)"
    echo "  --extract-text      Extract text from PDFs (local only, save to extracted_text/)"
    echo "  --summarize         Generate summaries (requires Ollama, for papers with PDFs)"
    echo "  --cleanup           Cleanup library (local files only, remove papers without PDFs and orphaned files)"
    echo "  --llm-operation     Advanced LLM operations (requires Ollama)"
    echo "  --test              Run test suite (with coverage analysis)"
    echo
    echo "Main Menu Options (0-9):"
    echo
    echo "Orchestrated Pipelines:"
    echo "  0  Full Pipeline (search + download + extract + summarize)"
    echo "  1  Meta-Analysis Pipeline (search + download + extract + meta-analysis)"
    echo
    echo "Individual Operations:"
    echo "  2  Search Only (network only - add to bibliography)"
    echo "  3  Download Only (network only - download PDFs)"
    echo "  4  Extract Text (local only - extract text from PDFs)"
    echo "  5  Summarize (requires Ollama - generate summaries)"
    echo "  6  Cleanup (local files only - remove papers without PDFs and orphaned files)"
    echo "  7  Advanced LLM Operations (requires Ollama)"
    echo
    echo "Testing:"
    echo "  9  Run Test Suite (with coverage analysis)"
    echo
    echo "  8  Exit"
    echo
    echo "Examples:"
    echo "  $0                      # Interactive menu mode"
    echo "  $0 --search            # Search literature (add to bibliography)"
    echo "  $0 --download          # Download PDFs (for bibliography entries)"
    echo "  $0 --extract-text      # Extract text from PDFs"
    echo "  $0 --summarize         # Generate summaries (for papers with PDFs)"
    echo "  $0 --cleanup           # Cleanup library (remove papers without PDFs and orphaned files)"
    echo "  $0 --test               # Run test suite (with coverage analysis)"
    echo "  $0 --option 0          # Run full pipeline (non-interactive)"
    echo "  $0 --option 0123       # Run sequence: 0, 1, 2, 3"
    echo
    echo "Environment Variables:"
    echo "  LOG_LEVEL             Logging verbosity (0=DEBUG, 1=INFO, 2=WARN, 3=ERROR)"
    echo "  PIPELINE_LOG_FILE      Optional log file path for file logging"
    echo
}

# ============================================================================
# Main Entry Point
# ============================================================================

# Function: main
# Purpose: Main entry point - parse arguments and run interactive menu or execute operation
# Args:
#   $@: Command-line arguments
# Returns:
#   Exits with operation exit code
# Side effects:
#   - Parses command-line arguments
#   - Executes operations or starts interactive menu
#   - Sets up environment
# Dependencies:
#   - All operation functions
#   - show_help()
# Example:
#   main "$@"
main() {
    local debug_mode=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h)
                show_help
                exit 0
                ;;
            --debug)
                debug_mode=true
                export LOG_LEVEL=0
                log_debug "Debug mode enabled"
                shift
                ;;
            --option)
                if [[ -z "${2:-}" ]]; then
                    log_error "Missing option number"
                    show_help
                    exit 1
                fi
                if [[ "$debug_mode" == "true" ]]; then
                    log_debug "Non-interactive mode with option: $2"
                fi
                run_non_interactive "$2"
                exit $?
                ;;
            --search)
                run_literature_search
                exit $?
                ;;
            --download)
                run_literature_download
                exit $?
                ;;
            --extract-text)
                run_literature_extract_text
                exit $?
                ;;
            --summarize)
                run_literature_summarize
                exit $?
                ;;
            --cleanup)
                run_literature_cleanup
                exit $?
                ;;
            --llm-operation)
                run_literature_llm_operations
                exit $?
                ;;
            --test)
                run_test_suite
                exit $?
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
        shift
    done
    
    # Interactive menu mode
    log_debug "Starting interactive menu mode"
    while true; do
        display_menu
        
        echo -n "Select option [0-9]: "
        read -r choice

        local exit_code=0
        if parse_choice_sequence "$choice" && [[ ${#SHORTHAND_CHOICES[@]} -gt 1 ]]; then
            run_option_sequence "${SHORTHAND_CHOICES[@]}"
            exit_code=$?
        else
            handle_menu_choice "$choice"
            exit_code=$?
        fi

        if [[ $exit_code -ne 0 ]]; then
            log_error_to_file "Last operation exited with code $exit_code"
            log_error "Last operation exited with code $exit_code"
        fi

        # Exit if choice is 8
        if [[ "$choice" == "8" ]]; then
            log_debug "Exit chosen, terminating"
            break
        fi

        # Don't prompt for cleanup option
        if [[ "$choice" != "6" ]]; then
            press_enter_to_continue
        fi
    done
}

# Run main
main "$@"
