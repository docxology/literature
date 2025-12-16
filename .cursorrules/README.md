# Cursor Rules

This directory contains modular rule files that guide code generation, development practices, and project conventions for the Literature Search and Management System.

## Overview

Cursor automatically loads all `.md` files from this `.cursorrules/` directory. Rules are organized by topic into separate files for better maintainability and clarity.

## Rule Files

### [general.md](general.md)
General coding practices and principles. Covers modular design, code organization, and professional development standards.

### [code-quality.md](code-quality.md)
Code quality standards and style guidelines. Defines expectations for professional, functional, intelligent, and elegant code.

### [testing.md](testing.md)
Testing guidelines and test-driven development (TDD) practices. Emphasizes real data analysis and unified modular testing approaches.

### [documentation.md](documentation.md)
Documentation requirements including AGENTS.md and README.md standards for every folder level. Covers documentation style and code commenting.

### [architecture.md](architecture.md)
Architecture and design principles. Guidelines for modular architecture, design patterns, and system organization.

### [project-specific.md](project-specific.md)
Literature search system specific conventions. Standalone repository requirements, bibliography management, and data directory structure.

## How Rules Are Applied

- All `.md` files in this directory are automatically loaded by Cursor
- Rules are cumulative - all files contribute to the complete rule set
- More specific rules (e.g., project-specific) complement general rules
- When rules conflict, more specific or later-loaded rules take precedence

## Best Practices

- Keep rules clear, actionable, and specific
- Use "show not tell" principle in rule descriptions
- Reference related rules in other files when appropriate
- Update rules as project conventions evolve
- Maintain consistency across all rule files






