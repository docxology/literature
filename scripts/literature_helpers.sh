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

# Function: clear_llm_context
# Purpose: Clear LLM client context to start fresh
# Args: None
# Returns:
#   0: Context cleared successfully
#   1: Failed to clear context
# Side effects:
#   - Executes Python code to clear LLM context
#   - Logs context clearing status
# Dependencies:
#   - infrastructure.llm module
#   - REPO_ROOT constant must be defined
# Example:
#   if clear_llm_context; then
#       echo "Context cleared"
#   fi
clear_llm_context() {
    log_info "Clearing LLM context..."
    
    if ! cd "$REPO_ROOT" 2>/dev/null; then
        log_error "Cannot change to repository root: $REPO_ROOT"
        return 1
    fi
    
    python3 << 'PYTHON_EOF'
import sys
import json
try:
    from infrastructure.llm import LLMClient, LLMConfig
    
    # Create client to access context
    config = LLMConfig.from_env()
    client = LLMClient(config=config)
    
    # Get context info before clearing
    messages_before = len(client.context.messages)
    tokens_before = client.context.estimated_tokens
    
    # Clear context
    client.reset()
    
    # Get context info after clearing
    messages_after = len(client.context.messages)
    tokens_after = client.context.estimated_tokens
    
    # Output JSON with status
    status = {
        "success": True,
        "messages_cleared": messages_before,
        "tokens_cleared": tokens_before,
        "messages_after": messages_after,
        "tokens_after": tokens_after
    }
    print(json.dumps(status))
    sys.exit(0)
except Exception as e:
    status = {
        "success": False,
        "error": str(e)
    }
    print(json.dumps(status))
    sys.exit(1)
PYTHON_EOF
    
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        log_debug "LLM context cleared successfully"
        return 0
    else
        log_error "Failed to clear LLM context"
        return 1
    fi
}

# Function: ensure_ollama_ready_with_restart
# Purpose: Ensure Ollama is ready, restarting if needed, and return comprehensive status
# Args: None
# Returns:
#   0: Ollama is ready
#   1: Ollama is not ready
# Side effects:
#   - Executes Python code to check/restart Ollama
#   - Outputs JSON status to stdout
#   - Logs comprehensive readiness information
# Dependencies:
#   - infrastructure.llm.utils.ollama module
#   - REPO_ROOT constant must be defined
# Example:
#   if ensure_ollama_ready_with_restart; then
#       echo "Ollama is ready"
#   fi
ensure_ollama_ready_with_restart() {
    log_info "Ensuring Ollama is ready (with restart if needed)..."
    
    if ! cd "$REPO_ROOT" 2>/dev/null; then
        log_error "Cannot change to repository root: $REPO_ROOT"
        return 1
    fi
    
    local status_json
    local python_stderr
    local stderr_file
    stderr_file=$(mktemp /tmp/ollama_check_stderr.XXXXXX 2>/dev/null || echo "/tmp/ollama_check_stderr.$$")
    # Capture stdout and stderr separately
    status_json=$(python3 << 'PYTHON_EOF' 2> "$stderr_file"
import sys
import json
import traceback
try:
    from infrastructure.llm.utils.ollama import (
        is_ollama_running,
        restart_ollama_server,
        get_model_names,
        ensure_ollama_ready,
        test_ollama_functionality,
        diagnose_ollama_issues
    )
    from infrastructure.llm import LLMClient, LLMConfig
    
    base_url = "http://localhost:11434"
    status = {
        "ready": False,
        "was_running": False,
        "restarted": False,
        "restart_status": None,
        "models": [],
        "model_count": 0,
        "connection_check": None,
        "functionality_test": None,
        "diagnostics": None,
        "error": None,
        "error_type": None
    }
    
    # Run comprehensive diagnostics first
    try:
        diag = diagnose_ollama_issues(base_url)
        status["diagnostics"] = diag
        
        if not diag["installed"]:
            status["error"] = "Ollama is not installed. Install from https://ollama.ai"
            status["error_type"] = "not_installed"
            print(json.dumps(status))
            sys.exit(1)
    except Exception as e:
        status["error"] = f"Failed to run diagnostics: {str(e)}"
        status["error_type"] = "diagnostics_error"
        print(json.dumps(status))
        sys.exit(1)
    
    # Check if already running
    was_running = is_ollama_running(base_url, timeout=2.0)
    status["was_running"] = was_running
    
    if not was_running:
        # Attempt restart
        restart_success, restart_msg = restart_ollama_server(
            base_url=base_url,
            kill_existing=True,
            wait_seconds=5.0
        )
        status["restarted"] = True
        status["restart_status"] = restart_msg
        
        if not restart_success:
            status["error"] = f"Ollama server not responding and restart failed: {restart_msg}"
            status["error_type"] = "restart_failed"
            print(json.dumps(status))
            sys.exit(1)
    
    # Verify Ollama is ready with models
    if not ensure_ollama_ready(base_url=base_url, auto_start=False, test_functionality=False):
        # Get more specific error from diagnostics
        if not status["diagnostics"]["server_running"]:
            status["error"] = "Ollama server is not responding. Start with: ollama serve"
            status["error_type"] = "server_not_running"
        elif not status["diagnostics"]["models_available"]:
            status["error"] = "Ollama running but no models available. Install with: ollama pull <model>"
            status["error_type"] = "no_models"
        else:
            status["error"] = "Ollama validation failed"
            status["error_type"] = "validation_failed"
        print(json.dumps(status))
        sys.exit(1)
    
    # Get model information
    models = get_model_names(base_url)
    status["models"] = models
    status["model_count"] = len(models)
    
    # Perform connection health check
    try:
        client = LLMClient(LLMConfig.from_env())
        is_available, error_msg = client.check_connection_detailed(timeout=3.0)
        status["connection_check"] = {
            "available": is_available,
            "error": error_msg
        }
        if not is_available:
            status["error"] = f"Connection check failed: {error_msg}"
            status["error_type"] = "connection_failed"
            print(json.dumps(status))
            sys.exit(1)
    except Exception as e:
        status["connection_check"] = {
            "available": False,
            "error": str(e)
        }
        status["error"] = f"Connection check error: {str(e)}"
        status["error_type"] = "connection_error"
        print(json.dumps(status))
        sys.exit(1)
    
    # Test functionality with actual query
    try:
        func_success, func_error = test_ollama_functionality(base_url, timeout=10.0)
        status["functionality_test"] = {
            "success": func_success,
            "error": func_error
        }
        if not func_success:
            status["error"] = f"Query test failed: {func_error}"
            status["error_type"] = "functionality_test_failed"
            print(json.dumps(status))
            sys.exit(1)
    except Exception as e:
        status["functionality_test"] = {
            "success": False,
            "error": str(e)
        }
        status["error"] = f"Functionality test error: {str(e)}"
        status["error_type"] = "functionality_test_error"
        print(json.dumps(status))
        sys.exit(1)
    
    status["ready"] = True
    print(json.dumps(status))
    sys.exit(0)
    
except ImportError as e:
    status = {
        "ready": False,
        "error": f"Import error: {str(e)}",
        "error_type": "import_error",
        "traceback": traceback.format_exc()
    }
    print(json.dumps(status))
    sys.exit(1)
except Exception as e:
    status = {
        "ready": False,
        "error": f"Unexpected error: {str(e)}",
        "error_type": "unexpected_error",
        "traceback": traceback.format_exc()
    }
    print(json.dumps(status))
    sys.exit(1)
PYTHON_EOF
    )
    
    local exit_code=$?
    python_stderr=$(cat "$stderr_file" 2>/dev/null || echo "")
    rm -f "$stderr_file"
    
    # Try to parse JSON from output (may be mixed with stderr)
    local json_output
    json_output=$(echo "$status_json" | grep -E '^\{.*\}$' | tail -1 || echo "")
    
    # If no JSON found, try parsing the whole output
    if [[ -z "$json_output" ]]; then
        json_output="$status_json"
    fi
    
    # Try to extract error message from JSON
    local error_msg=""
    local error_type=""
    local ready="false"
    
    if [[ -n "$json_output" ]]; then
        # Validate JSON and extract fields
        ready=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('ready', False))" 2>/dev/null || echo "false")
        error_msg=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('error', 'Unknown error'))" 2>/dev/null || echo "")
        error_type=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('error_type', ''))" 2>/dev/null || echo "")
    fi
    
    # If JSON parsing failed, show raw output for debugging
    if [[ -z "$error_msg" ]] || [[ "$error_msg" == "Unknown error" ]]; then
        if [[ -n "$python_stderr" ]]; then
            error_msg="Python script error (exit code $exit_code). Raw output: ${python_stderr:0:200}"
        else
            error_msg="Failed to parse Python script output (exit code $exit_code)"
        fi
    fi
    
    # Handle success case
    if [[ $exit_code -eq 0 ]] && [[ "$ready" == "True" ]]; then
        # Log comprehensive status
        local was_running
        was_running=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('was_running', False))" 2>/dev/null || echo "false")
        local restarted
        restarted=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('restarted', False))" 2>/dev/null || echo "false")
        local model_count
        model_count=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('model_count', 0))" 2>/dev/null || echo "0")
        local models_str
        models_str=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); models = d.get('models', []); print(', '.join(models[:5]))" 2>/dev/null || echo "")
        
        if [[ "$was_running" == "True" ]]; then
            log_success "Ollama is running and ready"
        else
            log_success "Ollama restarted and ready"
        fi
        
        log_info "  • Models available: $model_count"
        if [[ -n "$models_str" ]]; then
            log_info "  • Model names: $models_str"
        fi
        
        # Check connection health
        local conn_available
        conn_available=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); check = d.get('connection_check', {}); print(check.get('available', False))" 2>/dev/null || echo "false")
        if [[ "$conn_available" == "True" ]]; then
            log_info "  • Connection health: OK"
        else
            local conn_error
            conn_error=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); check = d.get('connection_check', {}); print(check.get('error', 'Unknown error'))" 2>/dev/null || echo "Unknown error")
            log_warning "  • Connection health: Warning - $conn_error"
        fi
        
        # Check functionality test
        local func_success
        func_success=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); test = d.get('functionality_test', {}); print(test.get('success', False))" 2>/dev/null || echo "false")
        if [[ "$func_success" == "True" ]]; then
            log_info "  • Functionality test: OK"
        else
            local func_error
            func_error=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); test = d.get('functionality_test', {}); print(test.get('error', 'Unknown error'))" 2>/dev/null || echo "Unknown error")
            log_warning "  • Functionality test: Warning - $func_error"
        fi
        
        return 0
    fi
    
    # Handle error cases with specific messages
    case "$error_type" in
        "not_installed")
            log_error "Ollama is not installed"
            log_info "  Install from: https://ollama.ai"
            ;;
        "server_not_running")
            log_error "Ollama server is not responding"
            log_info "  Start with: ollama serve"
            ;;
        "no_models")
            log_error "Ollama running but no models available"
            log_info "  Install a model with: ollama pull <model>"
            log_info "  Example: ollama pull llama3:latest"
            ;;
        "restart_failed")
            log_error "Failed to restart Ollama server"
            log_info "  Check if Ollama is installed: ollama --version"
            log_info "  Try starting manually: ollama serve"
            ;;
        "connection_failed"|"connection_error")
            log_error "Ollama connection check failed"
            ;;
        "functionality_test_failed"|"functionality_test_error")
            log_error "Ollama query test failed"
            log_info "  Server responds but cannot process queries"
            ;;
        "import_error")
            log_error "Failed to import required modules"
            log_info "  Check Python environment and dependencies"
            ;;
        *)
            log_error "Ollama is not ready: $error_msg"
            ;;
    esac
    
    # Show additional diagnostic info if available
    if [[ -n "$json_output" ]]; then
        local diag_installed
        diag_installed=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); diag = d.get('diagnostics', {}); print(diag.get('installed', False))" 2>/dev/null || echo "")
        if [[ "$diag_installed" == "False" ]] && [[ -z "$error_type" ]]; then
            log_info "  Diagnostic: Ollama may not be installed"
        fi
    fi
    
    return 1
}

# ============================================================================
# User Input Utilities
# ============================================================================

# Function: prompt_yes_no_default
# Purpose: Prompt user for yes/no with default value
# Args:
#   $1: Prompt text (will be displayed as "Prompt text? [Y/n]: " or "Prompt text? [y/N]: ")
#   $2: Default value ("Y" or "y" for yes, "N" or "n" for no, default: "Y")
# Returns:
#   0: User selected yes (or defaulted to yes)
#   1: User selected no (or defaulted to no)
# Side effects:
#   - Prompts user and reads input
#   - Handles empty input as default
# Example:
#   if prompt_yes_no_default "Run operation" "Y"; then
#       echo "Running operation"
#   fi
prompt_yes_no_default() {
    local prompt_text="${1:-}"
    local default="${2:-Y}"
    
    if [[ -z "$prompt_text" ]]; then
        log_error_with_context "prompt_yes_no_default: No prompt text provided"
        return 1
    fi
    
    # Normalize default to uppercase for display
    local default_upper
    default_upper=$(echo "$default" | tr '[:lower:]' '[:upper:]')
    
    # Determine prompt format based on default
    local prompt_format
    if [[ "$default_upper" == "Y" ]]; then
        prompt_format="[Y/n]"
    else
        prompt_format="[y/N]"
    fi
    
    # Display prompt and read input
    echo -n "${prompt_text}? ${prompt_format}: "
    read -r response
    
    # Normalize response to uppercase
    local response_upper
    response_upper=$(echo "$response" | tr '[:lower:]' '[:upper:]')
    
    # Handle empty input (use default)
    if [[ -z "$response_upper" ]]; then
        if [[ "$default_upper" == "Y" ]]; then
            return 0
        else
            return 1
        fi
    fi
    
    # Handle explicit yes/no
    if [[ "$response_upper" == "Y" ]] || [[ "$response_upper" == "YES" ]]; then
        return 0
    elif [[ "$response_upper" == "N" ]] || [[ "$response_upper" == "NO" ]]; then
        return 1
    else
        # Invalid input - use default
        log_warning "Invalid input '${response}', using default: ${default_upper}"
        if [[ "$default_upper" == "Y" ]]; then
            return 0
        else
            return 1
        fi
    fi
}

