#!/usr/bin/env bash
################################################################################
# Literature Errors Module
#
# Error handling utilities for literature operations:
# - Error context extraction
# - Contextual error logging
# - Safe command execution
# - Command output capture
#
# Dependencies:
#   - bash_utils.sh (must be sourced before this module)
################################################################################

# ============================================================================
# Error Context Utilities
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
# Side effects: Logs to both terminal and file
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

# ============================================================================
# Safe Execution Utilities
# ============================================================================

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
#   $1: Output file path for stdout
#   $2: Error file path for stderr (optional)
#   $@: Command and arguments to execute
# Returns:
#   Exit code of the command
# Side effects:
#   - Creates output files with command output
# Note: This function expects command as separate arguments, not a string
# Example: capture_command_output output.txt error.txt python3 script.py arg1 arg2
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

