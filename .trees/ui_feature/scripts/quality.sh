#!/bin/bash
# Run all quality checks: format, lint, and test

set -e

echo "🚀 Running full quality check pipeline..."

# Format code
./scripts/format.sh

# Run linting (will show remaining issues)
echo ""
echo "📋 Checking code quality..."
./scripts/lint.sh || echo "⚠️  Some linting issues remain (see above)"

# Run tests
echo ""
./scripts/test.sh

echo ""
echo "🎉 Quality check pipeline complete!"