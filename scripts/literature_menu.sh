#!/usr/bin/env bash
################################################################################
# Literature Menu Module
#
# Menu display and handling functions for interactive mode:
# - Library statistics display
# - Interactive menu rendering
# - Menu choice routing
# - Option sequence execution
#
# Dependencies:
#   - bash_utils.sh (must be sourced before this module)
#   - literature_helpers.sh (must be sourced before this module)
#   - literature_operations.sh (must be sourced before this module)
#   - literature_testing.sh (must be sourced before this module)
#   - REPO_ROOT constant (must be defined)
################################################################################

# ============================================================================
# Library Statistics Display
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
#   - REPO_ROOT constant must be defined
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

# ============================================================================
# Menu Display
# ============================================================================

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
#   - REPO_ROOT constant must be defined
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
    
    # Menu options - 7 major categories
    echo -e "${BOLD}1. Set Up Environment${NC}"
    echo -e "  1.1 Check dependencies (Python, Ollama, packages)"
    echo -e "  1.2 Validate configuration (env vars, file paths)"
    echo -e "  1.3 Create directories (data/, pdfs/, summaries/, etc.)"
    echo -e "  1.4 Check connectivity (network, Ollama server)"
    echo
    echo -e "${BOLD}2. Run Test Suite${NC}"
    echo -e "  2.1 Run test suite with coverage"
    echo
    echo -e "${BOLD}3. Search for Adding Citations (Network)${NC}"
    echo -e "  3.1 Search and add to bibliography"
    echo
    echo -e "${BOLD}4. PDF Download and Plaintext Extraction (Network)${NC}"
    echo -e "  4.1 Download PDFs only"
    echo -e "  4.2 Extract text only"
    echo -e "  4.3 Both (download then extract)"
    echo
    echo -e "${BOLD}5. LLM-Based Analysis (Ollama Local LLM)${NC}"
    echo -e "  5.1 LLM operations ${YELLOW}(prompts for: summarize, literature review, comparisons, etc.)${NC}"
    echo
    echo -e "${BOLD}6. Meta-Analysis of Library (No LLM)${NC}"
    echo -e "  6.1 Run meta-analysis ${CYAN}(bibliographic, citations, PCA, word use, source clarity, full text availability)${NC}"
    echo
    echo -e "${BOLD}7. Clear Library Entirely${NC}"
    echo -e "  7.1 Clear library ${RED}(removes all entries, PDFs, summaries, BibTeX, progress files)${NC}"
    echo
    echo "  0. Exit"
    echo
    
    # Footer with environment info
    echo -e "${BLUE}============================================================${NC}"
    echo -e "  Repository: ${CYAN}$REPO_ROOT${NC}"
    echo -e "  Python: ${CYAN}$(python3 --version 2>&1)${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo
}

# ============================================================================
# Menu Choice Handling
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
#   - All run_literature_* functions from literature_operations.sh
#   - run_test_suite() from literature_testing.sh
#   - validate_menu_choice() from literature_helpers.sh
# Example:
#   handle_menu_choice "0"
handle_menu_choice() {
    local choice="$1"
    local start_time end_time duration
    local exit_code=0
    
    # Handle exit
    if [[ "$choice" == "0" ]]; then
        log_debug "Exit chosen"
        return 0
    fi
    
    # Validate choice format (category.subcategory or single number for category 1 submenu)
    if [[ ! "$choice" =~ ^[1-7](\.[1-4])?$ ]]; then
        log_error "Invalid menu choice format: $choice"
        log_info "Please enter a valid option (e.g., 1.1, 4.3) or 0 to exit"
        return 1
    fi
    
    start_time=$(date +%s)
    log_debug "Handling menu choice: $choice"
    
    case "$choice" in
        # Category 1: Set Up Environment
        1.1)
            run_environment_check_dependencies
            exit_code=$?
            ;;
        1.2)
            run_environment_validate_config
            exit_code=$?
            ;;
        1.3)
            run_environment_create_directories
            exit_code=$?
            ;;
        1.4)
            run_environment_check_connectivity
            exit_code=$?
            ;;
        # Category 2: Run Test Suite
        2.1)
            run_test_suite
            exit_code=$?
            ;;
        # Category 3: Search for Adding Citations
        3.1)
            run_literature_search
            exit_code=$?
            ;;
        # Category 4: PDF Download and Plaintext Extraction
        4.1)
            run_literature_download
            exit_code=$?
            ;;
        4.2)
            run_literature_extract_text
            exit_code=$?
            ;;
        4.3)
            run_literature_download_and_extract
            exit_code=$?
            ;;
        # Category 5: LLM-Based Analysis
        5.1)
            run_literature_llm_operations
            exit_code=$?
            ;;
        # Category 6: Meta-Analysis
        6.1)
            run_literature_meta_analysis
            exit_code=$?
            ;;
        # Category 7: Clear Library
        7.1)
            run_clear_library
            exit_code=$?
            ;;
        *)
            log_error "Invalid option: $choice"
            log_info "Please enter a valid option (e.g., 1.1, 4.3) or 0 to exit"
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

