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
#   2. User selects operation (category.subcategory format, e.g., 1.1, 4.3)
#   3. Execute selected operation via Python script
#   4. Display results and return to menu
#
# Non-Interactive Mode:
#   1. Parse command-line arguments
#   2. Execute corresponding operation directly
#   3. Exit with appropriate code
#
# Major Categories:
#   1. Set Up Environment
#      1.1 Check dependencies (Python, Ollama, packages)
#      1.2 Validate configuration (env vars, file paths)
#      1.3 Create directories (data/, pdfs/, summaries/, etc.)
#      1.4 Check connectivity (network, Ollama server)
#   2. Run Test Suite
#      2.1 Run test suite with coverage
#   3. Search for Adding Citations (Network)
#      3.1 Search and add to bibliography
#   4. PDF Download and Plaintext Extraction (Network)
#      4.1 Download PDFs only
#      4.2 Extract text only
#      4.3 Both (download then extract)
#   5. LLM-Based Analysis (Ollama Local LLM)
#      5.1 LLM operations (prompts for: summarize, literature review, comparisons, etc.)
#   6. Meta-Analysis of Library
#      6.1 Run meta-analysis on existing library (bibliographic, citations, PCA, word use, source clarity, full text availability)
#      6.2 Run meta-analysis with embeddings on existing library (includes 6.1 + Ollama semantic analysis)
#   7. Clear Library Entirely
#      7.1 Clear library (removes all entries, PDFs, summaries, BibTeX, progress files)
#
# Note: All operations default to incremental/additive mode except category 7 (clear library)
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
# - Python script not found: Ensure scripts/literature_search.py exists
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

# ============================================================================
# Setup and Configuration
# ============================================================================

# Source shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
source "$SCRIPT_DIR/scripts/bash_utils.sh"

# Source module files in dependency order
source "$SCRIPT_DIR/scripts/literature_helpers.sh"
source "$SCRIPT_DIR/scripts/literature_errors.sh"
source "$SCRIPT_DIR/scripts/literature_operations.sh"
source "$SCRIPT_DIR/scripts/literature_testing.sh"
source "$SCRIPT_DIR/scripts/literature_menu.sh"
source "$SCRIPT_DIR/scripts/literature_cli.sh"

# ============================================================================
# Constants
# ============================================================================

readonly COVERAGE_THRESHOLD=60
readonly PYTHON_SCRIPT="scripts/literature_search.py"
readonly MAX_TEST_OUTPUT_LINES=20
readonly MAX_FAILED_TEST_LINES=20

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
            --clear-library)
                run_clear_library
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
            --setup-env)
                run_environment_check_dependencies
                local deps_code=$?
                echo
                run_environment_validate_config
                local config_code=$?
                echo
                run_environment_create_directories
                local dirs_code=$?
                echo
                run_environment_check_connectivity
                local conn_code=$?
                echo
                if [[ $deps_code -eq 0 ]] && [[ $config_code -eq 0 ]] && [[ $dirs_code -eq 0 ]]; then
                    log_success "Environment setup completed"
                    exit 0
                else
                    log_error "Environment setup completed with errors"
                    exit 1
                fi
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
        
        echo -n "Select option (e.g., 1.1, 4.3, or 0 to exit): "
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

        # Exit if choice is 0
        if [[ "$choice" == "0" ]]; then
            log_debug "Exit chosen, terminating"
            break
        fi

        # Don't prompt for clear library option (already has confirmation)
        if [[ "$choice" != "7.1" ]]; then
            press_enter_to_continue
        fi
    done
}

# Run main
main "$@"
