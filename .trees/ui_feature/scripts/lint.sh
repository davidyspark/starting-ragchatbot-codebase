#!/bin/bash
# Run linting checks

set -e

echo "🔍 Running linting checks with ruff..."
uv run ruff check backend/ main.py

echo "🎯 Linting complete!"