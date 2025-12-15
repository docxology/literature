#!/usr/bin/env bash
################################################################################
# Literature Helpers Module
#
# Core utility functions for literature operations:
# - Command execution with timeout
# - Signal handling
# - Environment validation
# - Python script execution
# - Ollama availability checks
#
# Dependencies:
#   - bash_utils.sh (must be sourced before this module)
#   - REPO_ROOT, PYTHON_SCRIPT constants (must be defined)
################################################################################

# ============================================================================
# Command Execution Utilities
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

# ============================================================================
# Signal Handling
# ============================================================================

# Function: setup_signal_handlers
# Purpose: Setup signal handlers for graceful shutdown
# Args:
#   $1: Operation name (optional, default: "operation")
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

# ============================================================================
# Validation Utilities
# ============================================================================

# Function: validate_menu_choice
# Purpose: Validate that a menu choice is in valid format (category.subcategory or 0 for exit)
# Args:
#   $1: Choice string to validate
# Returns:
#   0: Valid choice
#   1: Invalid choice
validate_menu_choice() {
    local choice="${1:-}"
    
    if [[ -z "$choice" ]]; then
        return 1
    fi
    
    # Allow 0 for exit
    if [[ "$choice" == "0" ]]; then
        return 0
    fi
    
    # Validate category.subcategory format (1.1 through 7.4)
    if [[ "$choice" =~ ^[1-7](\.[1-4])?$ ]]; then
        return 0
    fi
    
    return 1
}

# ============================================================================
# Environment Validation
# ============================================================================

# Function: check_environment
# Purpose: Validate environment setup (Python, scripts, directories)
# Args: None
# Returns:
#   0: Environment is valid
#   1: Environment validation failed
# Side effects: Logs validation results
# Dependencies:
#   - REPO_ROOT, PYTHON_SCRIPT constants must be defined
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
# Dependencies:
#   - REPO_ROOT, PYTHON_SCRIPT constants must be defined
check_python_script() {
    if [[ ! -f "$REPO_ROOT/$PYTHON_SCRIPT" ]]; then
        log_error_with_context "Python script not found: $PYTHON_SCRIPT"
        log_info "Expected location: $REPO_ROOT/$PYTHON_SCRIPT"
        return 1
    fi
    return 0
}

# ============================================================================
# Python Script Execution
# ============================================================================

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
# Dependencies:
#   - REPO_ROOT, PYTHON_SCRIPT constants must be defined
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
# Ollama Utilities
# ============================================================================

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
#   - REPO_ROOT constant must be defined
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

