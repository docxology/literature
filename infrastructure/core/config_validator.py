"""Configuration validation utilities.

Provides validation functions for environment variables and configuration objects.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from infrastructure.core.logging_utils import get_logger
from infrastructure.core.exceptions import ConfigurationError

logger = get_logger(__name__)


def validate_literature_config(config: Any) -> Tuple[bool, List[str]]:
    """Validate LiteratureConfig instance.
    
    Args:
        config: LiteratureConfig instance to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Validate numeric fields
    if config.default_limit < 1:
        errors.append("LITERATURE_DEFAULT_LIMIT must be >= 1")
    
    if config.max_results < 1:
        errors.append("LITERATURE_MAX_RESULTS must be >= 1")
    
    if config.arxiv_delay < 0:
        errors.append("LITERATURE_ARXIV_DELAY must be >= 0")
    
    if config.semanticscholar_delay < 0:
        errors.append("LITERATURE_SEMANTICSCHOLAR_DELAY must be >= 0")
    
    if config.retry_attempts < 0:
        errors.append("LITERATURE_RETRY_ATTEMPTS must be >= 0")
    
    if config.retry_delay < 0:
        errors.append("LITERATURE_RETRY_DELAY must be >= 0")
    
    if config.timeout <= 0:
        errors.append("LITERATURE_TIMEOUT must be > 0")
    
    if config.pdf_download_timeout <= 0:
        errors.append("LITERATURE_PDF_DOWNLOAD_TIMEOUT must be > 0")
    
    if config.download_retry_attempts < 0:
        errors.append("LITERATURE_DOWNLOAD_RETRY_ATTEMPTS must be >= 0")
    
    if config.download_retry_delay < 0:
        errors.append("LITERATURE_DOWNLOAD_RETRY_DELAY must be >= 0")
    
    if config.max_parallel_downloads < 1:
        errors.append("LITERATURE_MAX_PARALLEL_DOWNLOADS must be >= 1")
    
    if config.max_url_attempts_per_pdf < 1:
        errors.append("LITERATURE_MAX_URL_ATTEMPTS_PER_PDF must be >= 1")
    
    if config.max_fallback_strategies < 1:
        errors.append("LITERATURE_MAX_FALLBACK_STRATEGIES must be >= 1")
    
    # Validate paths
    if config.download_dir:
        try:
            download_path = Path(config.download_dir)
            if not download_path.parent.exists():
                errors.append(f"LITERATURE_DOWNLOAD_DIR parent directory does not exist: {download_path.parent}")
        except Exception as e:
            errors.append(f"Invalid LITERATURE_DOWNLOAD_DIR: {e}")
    
    if config.bibtex_file:
        try:
            bibtex_path = Path(config.bibtex_file)
            if not bibtex_path.parent.exists():
                errors.append(f"LITERATURE_BIBTEX_FILE parent directory does not exist: {bibtex_path.parent}")
        except Exception as e:
            errors.append(f"Invalid LITERATURE_BIBTEX_FILE: {e}")
    
    if config.library_index_file:
        try:
            library_path = Path(config.library_index_file)
            if not library_path.parent.exists():
                errors.append(f"LITERATURE_LIBRARY_INDEX parent directory does not exist: {library_path.parent}")
        except Exception as e:
            errors.append(f"Invalid LITERATURE_LIBRARY_INDEX: {e}")
    
    # Validate sources
    valid_sources = {
        "arxiv", "semanticscholar", "biorxiv", "pubmed", 
        "europepmc", "crossref", "openalex", "dblp", "unpaywall"
    }
    for source in config.sources:
        if source not in valid_sources:
            errors.append(f"Invalid source: {source}. Valid sources: {', '.join(sorted(valid_sources))}")
    
    # Validate Unpaywall configuration
    if config.use_unpaywall and not config.unpaywall_email:
        errors.append("UNPAYWALL_EMAIL is required when LITERATURE_USE_UNPAYWALL=true")
    
    if config.use_unpaywall and config.unpaywall_email:
        if "@" not in config.unpaywall_email:
            errors.append("UNPAYWALL_EMAIL must be a valid email address")
    
    return len(errors) == 0, errors


def validate_llm_config(config: Any) -> Tuple[bool, List[str]]:
    """Validate LLMConfig instance.
    
    Args:
        config: LLMConfig instance to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Validate numeric fields
    if config.temperature < 0 or config.temperature > 2:
        errors.append("LLM_TEMPERATURE must be between 0 and 2")
    
    if config.max_tokens < 1:
        errors.append("LLM_MAX_TOKENS must be >= 1")
    
    if config.context_window < 1:
        errors.append("LLM_CONTEXT_WINDOW must be >= 1")
    
    if config.timeout <= 0:
        errors.append("LLM_TIMEOUT must be > 0")
    
    if config.short_max_tokens < 1:
        errors.append("LLM short_max_tokens must be >= 1")
    
    if config.long_max_tokens < 1:
        errors.append("LLM long_max_tokens must be >= 1")
    
    if config.long_max_tokens < config.long_min_tokens:
        errors.append("LLM long_max_tokens must be >= long_min_tokens")
    
    # Validate URL
    if config.base_url:
        if not config.base_url.startswith(("http://", "https://")):
            errors.append("OLLAMA_HOST must start with http:// or https://")
    
    # Validate model name
    if not config.default_model:
        errors.append("OLLAMA_MODEL must be set")
    
    return len(errors) == 0, errors


def validate_environment_variables() -> Tuple[bool, List[str]]:
    """Validate environment variables for common issues.
    
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Check for common typos or issues
    env_vars = {
        "LITERATURE_DEFAULT_LIMIT": ("int", 1, None),
        "LITERATURE_MAX_RESULTS": ("int", 1, None),
        "LITERATURE_ARXIV_DELAY": ("float", 0, None),
        "LITERATURE_SEMANTICSCHOLAR_DELAY": ("float", 0, None),
        "LITERATURE_RETRY_ATTEMPTS": ("int", 0, None),
        "LITERATURE_RETRY_DELAY": ("float", 0, None),
        "LITERATURE_TIMEOUT": ("float", 0.1, None),
        "LITERATURE_PDF_DOWNLOAD_TIMEOUT": ("float", 0.1, None),
        "LITERATURE_DOWNLOAD_RETRY_ATTEMPTS": ("int", 0, None),
        "LITERATURE_DOWNLOAD_RETRY_DELAY": ("float", 0, None),
        "LITERATURE_MAX_PARALLEL_DOWNLOADS": ("int", 1, None),
        "LLM_TEMPERATURE": ("float", 0, 2),
        "LLM_MAX_TOKENS": ("int", 1, None),
        "LLM_CONTEXT_WINDOW": ("int", 1, None),
        "LLM_TIMEOUT": ("float", 0.1, None),
        "LLM_LONG_MAX_TOKENS": ("int", 1, None),
        "MAX_PARALLEL_SUMMARIES": ("int", 1, None),
        "LOG_LEVEL": ("int", 0, 3),
    }
    
    for var_name, (var_type, min_val, max_val) in env_vars.items():
        value = os.environ.get(var_name)
        if value is None:
            continue  # Optional variables are fine
        
        try:
            if var_type == "int":
                int_val = int(value)
                if int_val < min_val:
                    warnings.append(f"{var_name}={value} is below minimum {min_val}")
                if max_val is not None and int_val > max_val:
                    warnings.append(f"{var_name}={value} is above maximum {max_val}")
            elif var_type == "float":
                float_val = float(value)
                if float_val < min_val:
                    warnings.append(f"{var_name}={value} is below minimum {min_val}")
                if max_val is not None and float_val > max_val:
                    warnings.append(f"{var_name}={value} is above maximum {max_val}")
        except ValueError:
            warnings.append(f"{var_name}={value} is not a valid {var_type}")
    
    # Check boolean variables
    bool_vars = [
        "LITERATURE_USE_UNPAYWALL",
        "LITERATURE_USE_BROWSER_USER_AGENT",
        "LITERATURE_TWO_STAGE_ENABLED",
        "NO_EMOJI",
    ]
    
    for var_name in bool_vars:
        value = os.environ.get(var_name)
        if value is None:
            continue
        value_lower = value.lower()
        if value_lower not in ("true", "false", "1", "0", "yes", "no", ""):
            warnings.append(f"{var_name}={value} is not a valid boolean (use true/false/1/0/yes/no)")
    
    # Check required variables when conditions are met
    if os.environ.get("LITERATURE_USE_UNPAYWALL", "").lower() in ("true", "1", "yes"):
        if not os.environ.get("UNPAYWALL_EMAIL"):
            warnings.append("UNPAYWALL_EMAIL is required when LITERATURE_USE_UNPAYWALL=true")
    
    return len(warnings) == 0, warnings


def validate_and_log_config(config_type: str, config: Any) -> bool:
    """Validate configuration and log warnings/errors.
    
    Args:
        config_type: Type of config ("literature" or "llm")
        config: Configuration object to validate
        
    Returns:
        True if valid, False otherwise
    """
    if config_type == "literature":
        is_valid, errors = validate_literature_config(config)
    elif config_type == "llm":
        is_valid, errors = validate_llm_config(config)
    else:
        logger.warning(f"Unknown config type: {config_type}")
        return True
    
    if not is_valid:
        logger.error(f"Configuration validation failed for {config_type}:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    return True


def validate_on_startup() -> bool:
    """Validate configuration on startup and log warnings.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    # Validate environment variables
    env_valid, env_warnings = validate_environment_variables()
    
    if env_warnings:
        logger.warning("Configuration warnings detected:")
        for warning in env_warnings:
            logger.warning(f"  - {warning}")
    
    # Try to validate actual config objects if they can be imported
    try:
        from infrastructure.literature import LiteratureConfig
        lit_config = LiteratureConfig.from_env()
        if not validate_and_log_config("literature", lit_config):
            return False
    except Exception as e:
        logger.debug(f"Could not validate LiteratureConfig: {e}")
    
    try:
        from infrastructure.llm import LLMConfig
        llm_config = LLMConfig.from_env()
        if not validate_and_log_config("llm", llm_config):
            return False
    except Exception as e:
        logger.debug(f"Could not validate LLMConfig: {e}")
    
    return True




