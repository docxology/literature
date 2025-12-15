#!/usr/bin/env bash
################################################################################
# Literature Testing Module
#
# Test suite execution and result parsing functions:
# - Test result parsing from pytest output
# - Coverage result parsing
# - Test suite execution with coverage analysis
#
# Dependencies:
#   - bash_utils.sh (must be sourced before this module)
#   - literature_helpers.sh (must be sourced before this module)
#   - literature_errors.sh (must be sourced before this module)
#   - REPO_ROOT, COVERAGE_THRESHOLD, MAX_FAILED_TEST_LINES constants (must be defined)
################################################################################

# ============================================================================
# Test Result Parsing
# ============================================================================

# Function: parse_test_results
# Purpose: Parse pytest test results with improved regex patterns
# Args:
#   $1: Test output file path
# Returns: Sets global variables: total_tests, passed_tests, failed_tests, skipped_tests
#   0: Successfully parsed
#   1: Parse failed or file not found
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
#   0: Successfully parsed
#   1: Parse failed or file not found
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

# ============================================================================
# Test Suite Execution
# ============================================================================

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
#   - REPO_ROOT, COVERAGE_THRESHOLD, MAX_FAILED_TEST_LINES constants must be defined
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

