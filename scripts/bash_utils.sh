#!/usr/bin/env bash
################################################################################
# Shared Bash Utilities
#
# Common utilities for run_manuscript.sh and run_literature.sh:
# - Color codes and formatting
# - Logging functions
# - Utility functions (duration formatting, choice parsing, etc.)
# - Environment setup
#
# This file is sourced by both run_manuscript.sh and run_literature.sh
################################################################################

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m' # No Color

# Get script directory and repo root
# When sourced, BASH_SOURCE[1] is the script that sourced this file
# BASH_SOURCE[0] is this file (bash_utils.sh)
sourcing_script="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
SCRIPT_DIR="$(cd "$(dirname "$sourcing_script")" && pwd)"

# Determine repo root: if script is in root, use it; if in scripts/, go up one level
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    REPO_ROOT="$SCRIPT_DIR"
fi

# Export for subprocess use
export PROJECT_ROOT="$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/infrastructure:${REPO_ROOT}/project/src:${PYTHONPATH:-}"

# ============================================================================
# Logging Configuration
# ============================================================================

# LOG_LEVEL values: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR
# LOG_TIMESTAMP: set to "true" to enable timestamps (default: false)
# LOG_STRUCTURED: set to "true" to enable structured logging with function context (default: false, requires DEBUG)
readonly LOG_TIMESTAMP="${LOG_TIMESTAMP:-false}"
readonly LOG_STRUCTURED="${LOG_STRUCTURED:-false}"

# Maximum log file size before warning (100MB)
readonly MAX_LOG_FILE_SIZE=$((100 * 1024 * 1024))

# ============================================================================
# Internal Logging Utilities
# ============================================================================

# Function: _get_timestamp
# Purpose: Get formatted timestamp string
# Returns: Timestamp string or empty if timestamps disabled
_get_timestamp() {
    if [[ "$LOG_TIMESTAMP" == "true" ]]; then
        date '+%Y-%m-%d %H:%M:%S'
    fi
}

# Function: _get_log_context
# Purpose: Get structured logging context (function name, line number)
# Returns: Context string or empty if structured logging disabled
_get_log_context() {
    if [[ "$LOG_STRUCTURED" == "true" ]] && [[ "${LOG_LEVEL:-1}" == "0" ]]; then
        local func_name="${FUNCNAME[2]:-unknown}"
        local line_num="${BASH_LINENO[1]:-0}"
        echo "[${func_name}:${line_num}]"
    fi
}

# Function: _should_log_level
# Purpose: Check if message should be logged based on LOG_LEVEL
# Args:
#   $1: Message level (0=DEBUG, 1=INFO, 2=WARN, 3=ERROR)
# Returns: 0 if should log, 1 if should not
_should_log_level() {
    local msg_level="$1"
    local log_level="${LOG_LEVEL:-1}"
    
    # Always log ERROR and WARN regardless of level
    if [[ "$msg_level" -ge 2 ]]; then
        return 0
    fi
    
    # For DEBUG and INFO, check against configured level
    if [[ "$msg_level" -ge "$log_level" ]]; then
        return 0
    fi
    
    return 1
}

# Function: _strip_ansi_codes
# Purpose: Strip ANSI color codes from text (more robust regex)
# Args:
#   $1: Text with ANSI codes
# Returns: Text without ANSI codes
_strip_ansi_codes() {
    local text="$1"
    # Match ANSI escape sequences: ESC[ followed by numbers, semicolons, and ending with m
    echo "$text" | sed -E 's/\x1b\[[0-9;]*[a-zA-Z]//g'
}

# Function: _check_log_file_size
# Purpose: Check log file size and warn if too large
# Args:
#   $1: Log file path
_check_log_file_size() {
    local log_file="$1"
    if [[ -n "$log_file" ]] && [[ -f "$log_file" ]]; then
        local file_size
        file_size=$(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file" 2>/dev/null || echo "0")
        if [[ "$file_size" -gt $MAX_LOG_FILE_SIZE ]]; then
            echo "⚠ Warning: Log file exceeds ${MAX_LOG_FILE_SIZE} bytes. Consider rotating logs." >&2
        fi
    fi
}

# Function: _log_with_level
# Purpose: Internal unified logging function with level filtering and formatting
# Args:
#   $1: Message level (0=DEBUG, 1=INFO, 2=WARN, 3=ERROR)
#   $2: Message text
#   $3: Color code (optional)
#   $4: Prefix symbol (optional)
_log_with_level() {
    local level="$1"
    local message="${2:-}"
    local color="${3:-}"
    local prefix="${4:-}"
    
    # Validate inputs
    if [[ -z "$message" ]]; then
        return 0
    fi
    
    # Check if we should log at this level
    if ! _should_log_level "$level"; then
        return 0
    fi
    
    # Build log line components
    local timestamp
    timestamp=$(_get_timestamp)
    local context
    context=$(_get_log_context)
    
    # Format message
    local formatted_message="$message"
    if [[ -n "$context" ]]; then
        formatted_message="${context} ${message}"
    fi
    if [[ -n "$timestamp" ]]; then
        formatted_message="[${timestamp}] ${formatted_message}"
    fi
    if [[ -n "$prefix" ]]; then
        formatted_message="${prefix} ${formatted_message}"
    fi
    
    # Output with color if provided
    if [[ -n "$color" ]]; then
        echo -e "${color}${formatted_message}${NC}"
    else
        echo "$formatted_message"
    fi
}

# ============================================================================
# Public Logging Functions
# ============================================================================

log_header() {
    local message="${1:-}"
    if [[ -z "$message" ]]; then
        return 0
    fi
    
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  ${message}${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
}

log_stage() {
    # Display stage header with progress percentage and ETA.
    # Args:
    #   $1: Stage number (1-indexed)
    #   $2: Stage name
    #   $3: Total number of stages
    #   $4: Pipeline start time (optional, for ETA calculation)
    local stage_num="${1:-}"
    local stage_name="${2:-}"
    local total_stages="${3:-}"
    local pipeline_start="${4:-}"
    
    # Validate inputs
    if [[ -z "$stage_num" ]] || [[ -z "$stage_name" ]] || [[ -z "$total_stages" ]]; then
        return 1
    fi
    
    # Validate numeric inputs
    if ! [[ "$stage_num" =~ ^[0-9]+$ ]] || ! [[ "$total_stages" =~ ^[0-9]+$ ]]; then
        return 1
    fi
    
    echo
    local percentage=$((stage_num * 100 / total_stages))
    echo -e "${YELLOW}[${stage_num}/${total_stages}] ${stage_name} (${percentage}% complete)${NC}"
    
    # Calculate ETA if pipeline start time provided
    if [[ -n "$pipeline_start" ]] && [[ "$pipeline_start" =~ ^[0-9]+$ ]]; then
        local current_time
        current_time=$(date +%s)
        local elapsed=$((current_time - pipeline_start))
        if [[ $elapsed -gt 0 ]] && [[ $stage_num -gt 0 ]]; then
            local avg_time_per_stage=$((elapsed / stage_num))
            local remaining_stages=$((total_stages - stage_num))
            local eta=$((avg_time_per_stage * remaining_stages))
            echo -e "${CYAN}  Elapsed: $(format_duration "$elapsed") | ETA: $(format_duration "$eta")${NC}"
        fi
    fi
    
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

log_stage_start() {
    # Log stage start with consistent formatting.
    # Args:
    #   $1: Stage number
    #   $2: Stage name
    #   $3: Total number of stages
    local stage_num="${1:-}"
    local stage_name="${2:-}"
    local total_stages="${3:-}"
    
    if [[ -z "$stage_num" ]] || [[ -z "$stage_name" ]] || [[ -z "$total_stages" ]]; then
        return 1
    fi
    
    _log_with_level 1 "Starting Stage ${stage_num}/${total_stages}: ${stage_name}" "$BLUE" "▶"
}

log_stage_end() {
    # Log stage completion with consistent formatting.
    # Args:
    #   $1: Stage number
    #   $2: Stage name
    #   $3: Duration string
    local stage_num="${1:-}"
    local stage_name="${2:-}"
    local duration="${3:-}"
    
    if [[ -z "$stage_num" ]] || [[ -z "$stage_name" ]]; then
        return 1
    fi
    
    local message="Completed Stage ${stage_num}: ${stage_name}"
    if [[ -n "$duration" ]]; then
        message="${message} (${duration})"
    fi
    _log_with_level 1 "$message" "$GREEN" "✓"
}

log_success() {
    local message="${1:-}"
    _log_with_level 1 "$message" "$GREEN" "✓"
}

log_error() {
    local message="${1:-}"
    _log_with_level 3 "$message" "$RED" "✗"
}

log_info() {
    local message="${1:-}"
    if _should_log_level 1; then
        echo "  ${message}"
    fi
}

log_warning() {
    local message="${1:-}"
    _log_with_level 2 "$message" "$YELLOW" "⚠"
}

log_debug() {
    # Log debug message if LOG_LEVEL is 0 (DEBUG)
    # LOG_LEVEL: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR
    local message="${1:-}"
    _log_with_level 0 "$message" "$CYAN" "[DEBUG]"
}

# ============================================================================
# File Logging Functions
# ============================================================================

# Function: _write_to_log_file
# Purpose: Internal function to write message to log file with proper formatting
# Args:
#   $1: Message text (with ANSI codes)
#   $2: Log level (0=DEBUG, 1=INFO, 2=WARN, 3=ERROR)
_write_to_log_file() {
    local message="$1"
    local level="$2"
    local log_file="${PIPELINE_LOG_FILE:-}"
    
    if [[ -z "$log_file" ]]; then
        return 0
    fi
    
    # Check if we should log at this level
    if ! _should_log_level "$level"; then
        return 0
    fi
    
    # Check log file size
    _check_log_file_size "$log_file"
    
    # Strip ANSI codes and write to file
    local clean_message
    clean_message=$(_strip_ansi_codes "$message")
    
    # Ensure log file directory exists
    local log_dir
    log_dir=$(dirname "$log_file")
    if [[ -n "$log_dir" ]] && [[ ! -d "$log_dir" ]]; then
        mkdir -p "$log_dir" 2>/dev/null || return 1
    fi
    
    echo "$clean_message" >> "$log_file" 2>/dev/null || return 1
    return 0
}

# Function: log_to_file
# Purpose: Log to both terminal and log file
# Args:
#   $1: Message text
#   $2: Log level (optional, default: 1=INFO)
log_to_file() {
    local message="${1:-}"
    local level="${2:-1}"
    
    if [[ -z "$message" ]]; then
        return 0
    fi
    
    # Always log to terminal
    echo "$message"
    
    # Also log to file if log file is set
    _write_to_log_file "$message" "$level"
}

# Wrapper functions to log bash output to file
log_header_to_file() {
    local message="${1:-}"
    if [[ -z "$message" ]]; then
        return 0
    fi
    
    local log_file="${PIPELINE_LOG_FILE:-}"
    log_header "$message"
    
    if [[ -n "$log_file" ]]; then
        local timestamp
        timestamp=$(_get_timestamp)
        local header_line="============================================================"
        {
            if [[ -n "$timestamp" ]]; then
                echo "[${timestamp}] ${header_line}"
                echo "[${timestamp}]   ${message}"
                echo "[${timestamp}] ${header_line}"
            else
                echo "$header_line"
                echo "  ${message}"
                echo "$header_line"
            fi
        } >> "$log_file" 2>/dev/null || true
        _check_log_file_size "$log_file"
    fi
}

log_info_to_file() {
    local message="${1:-}"
    if [[ -z "$message" ]]; then
        return 0
    fi
    
    local log_file="${PIPELINE_LOG_FILE:-}"
    log_info "$message"
    
    if [[ -n "$log_file" ]]; then
        local timestamp
        timestamp=$(_get_timestamp)
        local log_line="  ${message}"
        if [[ -n "$timestamp" ]]; then
            log_line="[${timestamp}] ${log_line}"
        fi
        _write_to_log_file "$log_line" 1
    fi
}

log_success_to_file() {
    local message="${1:-}"
    if [[ -z "$message" ]]; then
        return 0
    fi
    
    local log_file="${PIPELINE_LOG_FILE:-}"
    local output
    output=$(log_success "$message")
    
    if [[ -n "$log_file" ]]; then
        _write_to_log_file "$output" 1
    fi
}

log_error_to_file() {
    local message="${1:-}"
    if [[ -z "$message" ]]; then
        return 0
    fi
    
    local log_file="${PIPELINE_LOG_FILE:-}"
    local output
    output=$(log_error "$message")
    
    if [[ -n "$log_file" ]]; then
        _write_to_log_file "$output" 3
    fi
}

log_warning_to_file() {
    local message="${1:-}"
    if [[ -z "$message" ]]; then
        return 0
    fi
    
    local log_file="${PIPELINE_LOG_FILE:-}"
    local output
    output=$(log_warning "$message")
    
    if [[ -n "$log_file" ]]; then
        _write_to_log_file "$output" 2
    fi
}

log_debug_to_file() {
    # Log debug message to file if LOG_LEVEL is 0 (DEBUG)
    # Args:
    #   $1: Message text
    local message="${1:-}"
    if [[ -z "$message" ]]; then
        return 0
    fi
    
    local log_file="${PIPELINE_LOG_FILE:-}"
    local output
    output=$(log_debug "$message")
    
    if [[ -n "$log_file" ]] && [[ -n "$output" ]]; then
        _write_to_log_file "$output" 0
    fi
}

# ============================================================================
# Validation Functions
# ============================================================================

# Function: validate_numeric
# Purpose: Validate that input is a positive integer
# Args:
#   $1: Value to validate
#   $2: Minimum value (optional, default: 0)
#   $3: Maximum value (optional, default: no limit)
# Returns: 0 if valid, 1 if invalid
validate_numeric() {
    local value="${1:-}"
    local min="${2:-0}"
    local max="${3:-}"
    
    # Check if empty
    if [[ -z "$value" ]]; then
        return 1
    fi
    
    # Check if numeric
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        return 1
    fi
    
    # Check minimum
    if [[ "$value" -lt "$min" ]]; then
        return 1
    fi
    
    # Check maximum if provided
    if [[ -n "$max" ]] && [[ "$value" -gt "$max" ]]; then
        return 1
    fi
    
    return 0
}

# Function: safe_string
# Purpose: Sanitize string for safe use (remove control characters)
# Args:
#   $1: String to sanitize
# Returns: Sanitized string
safe_string() {
    local str="${1:-}"
    # Remove control characters except newline and tab
    echo "$str" | tr -d '\000-\010\013\014\016-\037'
}

# ============================================================================
# Utility Functions
# ============================================================================

get_elapsed_time() {
    local start_time="${1:-}"
    local end_time="${2:-}"
    
    # Validate inputs
    if ! validate_numeric "$start_time" || ! validate_numeric "$end_time"; then
        echo "0"
        return 1
    fi
    
    # Calculate elapsed time
    local elapsed=$((end_time - start_time))
    
    # Ensure non-negative
    if [[ $elapsed -lt 0 ]]; then
        echo "0"
        return 1
    fi
    
    echo "$elapsed"
    return 0
}

format_duration() {
    local seconds="${1:-0}"
    
    # Validate input
    if ! validate_numeric "$seconds" 0; then
        echo "0s"
        return 1
    fi
    
    # Handle edge cases
    if [[ $seconds -eq 0 ]]; then
        echo "0s"
        return 0
    fi
    
    # Less than a minute
    if (( seconds < 60 )); then
        echo "${seconds}s"
        return 0
    fi
    
    # Less than an hour
    if (( seconds < 3600 )); then
        local minutes=$((seconds / 60))
        local secs=$((seconds % 60))
        if [[ $secs -eq 0 ]]; then
            echo "${minutes}m"
        else
            echo "${minutes}m ${secs}s"
        fi
        return 0
    fi
    
    # Less than a day
    if (( seconds < 86400 )); then
        local hours=$((seconds / 3600))
        local minutes=$(((seconds % 3600) / 60))
        local secs=$((seconds % 60))
        local result="${hours}h"
        if [[ $minutes -gt 0 ]]; then
            result="${result} ${minutes}m"
        fi
        if [[ $secs -gt 0 ]] && [[ $minutes -eq 0 ]]; then
            result="${result} ${secs}s"
        fi
        echo "$result"
        return 0
    fi
    
    # Days and beyond
    local days=$((seconds / 86400))
    local hours=$(((seconds % 86400) / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local result="${days}d"
    if [[ $hours -gt 0 ]]; then
        result="${result} ${hours}h"
    fi
    if [[ $minutes -gt 0 ]] && [[ $hours -eq 0 ]]; then
        result="${result} ${minutes}m"
    fi
    echo "$result"
    return 0
}

press_enter_to_continue() {
    echo
    echo -e "${CYAN}Press Enter to return to menu...${NC}"
    read -r
}

# Parsed shorthand choices holder (for sequences like "0123" or "345")
SHORTHAND_CHOICES=()

# Function: parse_choice_sequence
# Purpose: Parse a user-supplied option string into a sequence of menu choices
# Supports:
# - Concatenated digits (e.g., "01234" or "345") → each digit is a choice
# - Comma/space separated numbers (e.g., "3,4,5" or "3 4 5")
# Args:
#   $1: Input string to parse
#   $2: Maximum valid choice (optional, for bounds checking)
# Returns: 0 on success and populates SHORTHAND_CHOICES; 1 on parse failure
parse_choice_sequence() {
    local input="${1:-}"
    local max_choice="${2:-9}"
    
    SHORTHAND_CHOICES=()
    
    # Validate input
    if [[ -z "$input" ]]; then
        return 1
    fi
    
    # Sanitize input (remove spaces)
    input="${input//[[:space:]]/}"
    
    # Validate max_choice
    if ! validate_numeric "$max_choice" 0; then
        max_choice=9
    fi
    
    # Pure digits with length > 1 → treat as shorthand digits
    if [[ "$input" =~ ^[0-9]+$ ]] && [[ ${#input} -gt 1 ]]; then
        for ((i = 0; i < ${#input}; i++)); do
            local digit="${input:i:1}"
            # Bounds check
            if validate_numeric "$digit" 0 "$max_choice"; then
                SHORTHAND_CHOICES+=("$digit")
            else
                log_debug "Invalid choice digit: $digit (max: $max_choice)"
                return 1
            fi
        done
        return 0
    fi
    
    # Otherwise split on commas
    IFS=',' read -ra parts <<< "$input"
    for part in "${parts[@]}"; do
        [[ -z "$part" ]] && continue
        
        # Validate numeric and bounds
        if validate_numeric "$part" 0 "$max_choice"; then
            SHORTHAND_CHOICES+=("$part")
        else
            log_debug "Invalid choice: $part (max: $max_choice)"
            return 1
        fi
    done
    
    if [[ ${#SHORTHAND_CHOICES[@]} -gt 0 ]]; then
        return 0
    fi
    
    return 1
}


