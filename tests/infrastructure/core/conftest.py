"""Core module pytest configuration and shared fixtures.

Provides fixtures specific to infrastructure/core tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@pytest.fixture
def sample_config_file(tmp_path):
    """Create a sample configuration file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
test_key: test_value
another_key: another_value
""")
    return config_file


@pytest.fixture
def sample_log_file(tmp_path):
    """Create a sample log file path."""
    log_file = tmp_path / "test.log"
    return log_file




