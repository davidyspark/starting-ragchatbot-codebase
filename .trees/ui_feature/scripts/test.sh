#!/bin/bash
# Run tests

set -e

echo "🧪 Running tests with pytest..."
cd backend && uv run pytest tests/ -v

echo "✅ Tests complete!"