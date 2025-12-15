#!/usr/bin/env bash
################################################################################
# Literature CLI Module
#
# CLI and help functions for non-interactive mode:
# - Help message display
# - Non-interactive mode handler
#
# Dependencies:
#   - bash_utils.sh (must be sourced before this module)
#   - literature_menu.sh (must be sourced before this module)
#   - literature_operations.sh (must be sourced before this module)
#   - parse_choice_sequence() from bash_utils.sh
################################################################################

# ============================================================================
# Help Functions
# ============================================================================

# Function: show_help
# Purpose: Display comprehensive help message with usage examples
# Args: None
# Returns: None
# Side effects:
#   - Outputs help text to stdout
# Note: Uses $0 which should be the main script name
# Example:
#   show_help
show_help() {
    local script_name="${0:-run_literature.sh}"
    echo "Usage: $script_name [OPTIONS]"
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
    echo "  --clear-library     Clear library entirely (removes all entries, PDFs, summaries, etc.)"
    echo "  --llm-operation     Advanced LLM operations (requires Ollama)"
    echo "  --test              Run test suite (with coverage analysis)"
    echo "  --setup-env         Set up environment (check dependencies, validate config, create dirs, check connectivity)"
    echo
    echo "Main Menu Options (category.subcategory format):"
    echo
    echo "1. Set Up Environment:"
    echo "  1.1  Check dependencies (Python, Ollama, packages)"
    echo "  1.2  Validate configuration (env vars, file paths)"
    echo "  1.3  Create directories (data/, pdfs/, summaries/, etc.)"
    echo "  1.4  Check connectivity (network, Ollama server)"
    echo
    echo "2. Run Test Suite:"
    echo "  2.1  Run test suite with coverage"
    echo
    echo "3. Search for Adding Citations (Network):"
    echo "  3.1  Search and add to bibliography"
    echo
    echo "4. PDF Download and Plaintext Extraction (Network):"
    echo "  4.1  Download PDFs only"
    echo "  4.2  Extract text only"
    echo "  4.3  Both (download then extract)"
    echo
    echo "5. LLM-Based Analysis (Ollama Local LLM):"
    echo "  5.1  LLM operations (prompts for: summarize, literature review, comparisons, etc.)"
    echo
    echo "6. Meta-Analysis of Library:"
    echo "  6.1  Run meta-analysis on existing library (bibliographic, citations, PCA, word use, etc.)"
    echo "  6.2  Run meta-analysis with embeddings on existing library (includes 6.1 + Ollama semantic analysis)"
    echo
    echo "7. Clear Library Entirely:"
    echo "  7.1  Clear library (removes all entries, PDFs, summaries, BibTeX, progress files)"
    echo
    echo "  0  Exit"
    echo
    echo "Examples:"
    echo "  $script_name                      # Interactive menu mode"
    echo "  $script_name --search            # Search literature (add to bibliography)"
    echo "  $script_name --download          # Download PDFs (for bibliography entries)"
    echo "  $script_name --extract-text      # Extract text from PDFs"
    echo "  $script_name --summarize         # Generate summaries (for papers with PDFs)"
    echo "  $script_name --clear-library     # Clear library entirely (removes all entries, PDFs, summaries, etc.)"
    echo "  $script_name --test              # Run test suite (with coverage analysis)"
    echo "  $script_name --setup-env         # Set up environment (check dependencies, validate config, create dirs, check connectivity)"
    echo "  $script_name --option 1.1        # Run option 1.1 (check dependencies) non-interactive"
    echo "  $script_name --option 4.3        # Run option 4.3 (download and extract) non-interactive"
    echo
    echo "Environment Variables:"
    echo "  LOG_LEVEL             Logging verbosity (0=DEBUG, 1=INFO, 2=WARN, 3=ERROR)"
    echo "  PIPELINE_LOG_FILE      Optional log file path for file logging"
    echo
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
#   - parse_choice_sequence() from bash_utils.sh
#   - run_option_sequence() from literature_menu.sh
#   - handle_menu_choice() from literature_menu.sh
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

