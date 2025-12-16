"""Root-level pytest configuration and shared fixtures.

Provides fixtures available to all tests in the test suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def repo_root():
    """Get repository root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_dir(repo_root):
    """Get test data directory."""
    test_data = repo_root / "tests" / "test_data"
    test_data.mkdir(parents=True, exist_ok=True)
    return test_data


@pytest.fixture
def temp_dir(tmp_path):
    """Get temporary directory for tests."""
    return tmp_path






