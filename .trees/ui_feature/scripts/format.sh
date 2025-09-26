#!/bin/bash
# Format code with black and fix imports with ruff

set -e

echo "🎨 Formatting Python code with black..."
uv run black backend/ main.py

echo "🔧 Fixing imports and auto-fixable issues with ruff..."
uv run ruff check --fix backend/ main.py

echo "✨ Code formatting complete!"