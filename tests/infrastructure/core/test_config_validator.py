"""Tests for configuration validation utilities."""
from __future__ import annotations

import os
import pytest
from pathlib import Path

from infrastructure.core.config_validator import (
    validate_literature_config,
    validate_llm_config,
    validate_environment_variables,
    validate_and_log_config,
    validate_on_startup,
)
from infrastructure.literature import LiteratureConfig
from infrastructure.llm import LLMConfig


class TestValidateLiteratureConfig:
    """Test literature configuration validation."""
    
    def test_valid_config(self):
        """Test validation of valid configuration."""
        config = LiteratureConfig(
            default_limit=25,
            max_results=100,
            timeout=30.0,
            pdf_download_timeout=60.0,
            sources=["arxiv", "semanticscholar"],
            use_unpaywall=False,
        )
        is_valid, errors = validate_literature_config(config)
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_default_limit(self):
        """Test validation fails for invalid default_limit."""
        config = LiteratureConfig(default_limit=0)
        is_valid, errors = validate_literature_config(config)
        assert not is_valid
        assert any("LITERATURE_DEFAULT_LIMIT" in e for e in errors)
    
    def test_invalid_timeout(self):
        """Test validation fails for invalid timeout."""
        config = LiteratureConfig(timeout=0)
        is_valid, errors = validate_literature_config(config)
        assert not is_valid
        assert any("LITERATURE_TIMEOUT" in e for e in errors)
    
    def test_invalid_source(self):
        """Test validation fails for invalid source."""
        config = LiteratureConfig(sources=["invalid_source"])
        is_valid, errors = validate_literature_config(config)
        assert not is_valid
        assert any("Invalid source" in e for e in errors)
    
    def test_unpaywall_without_email(self):
        """Test validation fails when Unpaywall enabled without email."""
        config = LiteratureConfig(use_unpaywall=True, unpaywall_email="")
        is_valid, errors = validate_literature_config(config)
        assert not is_valid
        assert any("UNPAYWALL_EMAIL" in e for e in errors)
    
    def test_unpaywall_invalid_email(self):
        """Test validation fails for invalid email."""
        config = LiteratureConfig(use_unpaywall=True, unpaywall_email="not-an-email")
        is_valid, errors = validate_literature_config(config)
        assert not is_valid
        assert any("email address" in e.lower() for e in errors)


class TestValidateLLMConfig:
    """Test LLM configuration validation."""
    
    def test_valid_config(self):
        """Test validation of valid configuration."""
        config = LLMConfig(
            temperature=0.7,
            max_tokens=2048,
            context_window=131072,
            timeout=60.0,
            default_model="gemma3:4b",
        )
        is_valid, errors = validate_llm_config(config)
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_temperature(self):
        """Test validation fails for invalid temperature."""
        config = LLMConfig(temperature=3.0)
        is_valid, errors = validate_llm_config(config)
        assert not is_valid
        assert any("LLM_TEMPERATURE" in e for e in errors)
    
    def test_invalid_max_tokens(self):
        """Test validation fails for invalid max_tokens."""
        config = LLMConfig(max_tokens=0)
        is_valid, errors = validate_llm_config(config)
        assert not is_valid
        assert any("LLM_MAX_TOKENS" in e for e in errors)
    
    def test_invalid_timeout(self):
        """Test validation fails for invalid timeout."""
        config = LLMConfig(timeout=0)
        is_valid, errors = validate_llm_config(config)
        assert not is_valid
        assert any("LLM_TIMEOUT" in e for e in errors)
    
    def test_invalid_base_url(self):
        """Test validation fails for invalid base_url."""
        config = LLMConfig(base_url="not-a-url")
        is_valid, errors = validate_llm_config(config)
        assert not is_valid
        assert any("OLLAMA_HOST" in e for e in errors)
    
    def test_missing_model(self):
        """Test validation fails for missing model."""
        config = LLMConfig(default_model="")
        is_valid, errors = validate_llm_config(config)
        assert not is_valid
        assert any("OLLAMA_MODEL" in e for e in errors)


class TestValidateEnvironmentVariables:
    """Test environment variable validation."""
    
    def test_valid_environment(self):
        """Test validation with valid environment variables."""
        # Set some valid values
        os.environ["LITERATURE_DEFAULT_LIMIT"] = "25"
        os.environ["LLM_TEMPERATURE"] = "0.7"
        os.environ["LOG_LEVEL"] = "1"
        
        try:
            is_valid, warnings = validate_environment_variables()
            # Should be valid (warnings are non-blocking)
            assert isinstance(is_valid, bool)
            assert isinstance(warnings, list)
        finally:
            # Clean up
            os.environ.pop("LITERATURE_DEFAULT_LIMIT", None)
            os.environ.pop("LLM_TEMPERATURE", None)
            os.environ.pop("LOG_LEVEL", None)
    
    def test_invalid_numeric_value(self):
        """Test validation detects invalid numeric values."""
        os.environ["LITERATURE_DEFAULT_LIMIT"] = "not-a-number"
        
        try:
            is_valid, warnings = validate_environment_variables()
            assert any("LITERATURE_DEFAULT_LIMIT" in w for w in warnings)
        finally:
            os.environ.pop("LITERATURE_DEFAULT_LIMIT", None)
    
    def test_invalid_boolean_value(self):
        """Test validation detects invalid boolean values."""
        os.environ["LITERATURE_USE_UNPAYWALL"] = "maybe"
        
        try:
            is_valid, warnings = validate_environment_variables()
            assert any("LITERATURE_USE_UNPAYWALL" in w for w in warnings)
        finally:
            os.environ.pop("LITERATURE_USE_UNPAYWALL", None)
    
    def test_unpaywall_requires_email(self):
        """Test validation warns when Unpaywall enabled without email."""
        os.environ["LITERATURE_USE_UNPAYWALL"] = "true"
        os.environ.pop("UNPAYWALL_EMAIL", None)  # Ensure not set
        
        try:
            is_valid, warnings = validate_environment_variables()
            assert any("UNPAYWALL_EMAIL" in w for w in warnings)
        finally:
            os.environ.pop("LITERATURE_USE_UNPAYWALL", None)


class TestValidateAndLogConfig:
    """Test validate_and_log_config function."""
    
    def test_valid_literature_config(self):
        """Test validation and logging for valid literature config."""
        config = LiteratureConfig()
        result = validate_and_log_config("literature", config)
        assert result is True
    
    def test_invalid_literature_config(self):
        """Test validation and logging for invalid literature config."""
        config = LiteratureConfig(default_limit=0)
        result = validate_and_log_config("literature", config)
        assert result is False
    
    def test_valid_llm_config(self):
        """Test validation and logging for valid LLM config."""
        config = LLMConfig()
        result = validate_and_log_config("llm", config)
        assert result is True
    
    def test_invalid_llm_config(self):
        """Test validation and logging for invalid LLM config."""
        config = LLMConfig(temperature=3.0)
        result = validate_and_log_config("llm", config)
        assert result is False
    
    def test_unknown_config_type(self):
        """Test validation handles unknown config type."""
        config = LiteratureConfig()
        result = validate_and_log_config("unknown", config)
        # Should return True (graceful handling)
        assert result is True


class TestValidateOnStartup:
    """Test validate_on_startup function."""
    
    def test_startup_validation(self):
        """Test startup validation runs without errors."""
        # Should not raise exceptions
        result = validate_on_startup()
        assert isinstance(result, bool)



