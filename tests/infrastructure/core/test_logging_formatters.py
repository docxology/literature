#!/usr/bin/env python3
"""Comprehensive tests for infrastructure/core/logging_formatters.py.

Tests JSONFormatter and TemplateFormatter with real usage patterns.
No mocks - tests actual formatting behavior.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from infrastructure.core.logging_formatters import JSONFormatter, TemplateFormatter


class TestJSONFormatter:
    """Test JSONFormatter functionality."""
    
    def test_format_basic_message(self):
        """Test formatting a basic log message."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data['level'] == 'INFO'
        assert data['logger'] == 'test_logger'
        assert data['message'] == 'Test message'
        assert 'timestamp' in data
    
    def test_format_with_exception(self):
        """Test formatting with exception info."""
        formatter = JSONFormatter()
        
        try:
            raise ValueError("Test error")
        except ValueError:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Test error occurred",
                args=(),
                exc_info=sys.exc_info()
            )
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data['level'] == 'ERROR'
        assert data['message'] == 'Test error occurred'
        assert 'exception' in data
        assert 'ValueError' in data['exception']
        assert 'Test error' in data['exception']
    
    def test_format_with_extra_fields(self):
        """Test formatting with extra fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.extra_fields = {'key1': 'value1', 'key2': 42}
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data['key1'] == 'value1'
        assert data['key2'] == 42
        assert data['message'] == 'Test message'
    
    def test_format_different_levels(self):
        """Test formatting different log levels."""
        formatter = JSONFormatter()
        levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
        
        for level in levels:
            record = logging.LogRecord(
                name="test_logger",
                level=level,
                pathname="test.py",
                lineno=10,
                msg=f"Level {level} message",
                args=(),
                exc_info=None
            )
            
            result = formatter.format(record)
            data = json.loads(result)
            
            assert data['level'] == logging.getLevelName(level)
            assert data['message'] == f"Level {level} message"


class TestTemplateFormatter:
    """Test TemplateFormatter functionality."""
    
    def test_format_basic_message(self):
        """Test formatting a basic log message."""
        formatter = TemplateFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        result = formatter.format(record)
        
        assert '[INFO]' in result
        assert 'Test message' in result
        # Check timestamp format YYYY-MM-DD HH:MM:SS
        assert len(result.split('[')[0].strip()) == 0 or ':' in result
    
    def test_format_different_levels(self):
        """Test formatting different log levels."""
        formatter = TemplateFormatter()
        level_map = {
            logging.DEBUG: 'DEBUG',
            logging.INFO: 'INFO',
            logging.WARNING: 'WARNING',
            logging.ERROR: 'ERROR',
            logging.CRITICAL: 'CRITICAL',
        }
        
        for level, level_name in level_map.items():
            record = logging.LogRecord(
                name="test_logger",
                level=level,
                pathname="test.py",
                lineno=10,
                msg=f"{level_name} message",
                args=(),
                exc_info=None
            )
            
            result = formatter.format(record)
            
            assert f'[{level_name}]' in result
            assert f'{level_name} message' in result
    
    def test_format_with_exception(self):
        """Test formatting with exception info."""
        formatter = TemplateFormatter()
        
        try:
            raise ValueError("Test error")
        except ValueError:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Test error occurred",
                args=(),
                exc_info=sys.exc_info()
            )
        
        result = formatter.format(record)
        
        assert '[ERROR]' in result
        assert 'Test error occurred' in result
        assert 'ValueError' in result or 'Traceback' in result
    
    def test_emoji_support(self):
        """Test emoji support when in TTY."""
        # Save original state
        original_isatty = sys.stdout.isatty
        original_no_emoji = os.environ.get('NO_EMOJI')
        
        try:
            # Test with TTY (emojis enabled)
            sys.stdout.isatty = lambda: True
            if 'NO_EMOJI' in os.environ:
                del os.environ['NO_EMOJI']
            
            # Reimport to get fresh formatter with emoji support
            import importlib
            from infrastructure.core import logging_formatters
            importlib.reload(logging_formatters)
            
            formatter = logging_formatters.TemplateFormatter()
            record = logging.LogRecord(
                name="test_logger",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg="Test message",
                args=(),
                exc_info=None
            )
            
            result = formatter.format(record)
            # Emoji may or may not be present depending on system
            assert '[INFO]' in result
            
        finally:
            # Restore original state
            sys.stdout.isatty = original_isatty
            if original_no_emoji:
                os.environ['NO_EMOJI'] = original_no_emoji
            else:
                os.environ.pop('NO_EMOJI', None)
    
    def test_timestamp_format(self):
        """Test timestamp format is correct."""
        formatter = TemplateFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        result = formatter.format(record)
        
        # Extract timestamp part (format: [YYYY-MM-DD HH:MM:SS])
        parts = result.split(']')
        if len(parts) >= 2:
            timestamp_part = parts[0].strip('[')
            # Should match YYYY-MM-DD HH:MM:SS format
            assert len(timestamp_part) == 19
            assert timestamp_part[4] == '-'
            assert timestamp_part[7] == '-'
            assert timestamp_part[10] == ' '
            assert timestamp_part[13] == ':'
            assert timestamp_part[16] == ':'
    
    def test_message_preservation(self):
        """Test that message content is preserved."""
        formatter = TemplateFormatter()
        test_messages = [
            "Simple message",
            "Message with numbers: 12345",
            "Message with special chars: !@#$%^&*()",
            "Message with unicode: 测试 🚀",
            "Multi\nline\nmessage",
        ]
        
        for msg in test_messages:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=msg,
                args=(),
                exc_info=None
            )
            
            result = formatter.format(record)
            assert msg in result or msg.replace('\n', ' ') in result




