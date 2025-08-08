#!/bin/bash

# Setup script for pre-commit hooks
set -e

echo "Setting up pre-commit hooks for the volumetric display project..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is required but not installed."
    exit 1
fi

# Note: All tools are managed by pre-commit environments, no local installations required

# Install pre-commit (not managed through Bazel requirements)
echo "Installing pre-commit..."
pip3 install pre-commit

# Install the git hooks
echo "Installing git hooks..."
pre-commit install

# Run pre-commit on all files to ensure everything is properly formatted
echo "Running pre-commit on all files..."
pre-commit run --all-files

echo "Pre-commit setup complete!"
echo ""
echo "The following hooks are now active:"
echo "- Python: Black (formatting), isort (imports), flake8 (linting)"
echo "- Rust: rustfmt (formatting), clippy (linting)"
echo "- C/C++: clang-format (formatting)"
echo "- Shell: shellcheck (linting)"
echo "- Bazel: buildifier (BUILD files)"
echo "- Nix: nixpkgs-fmt (formatting)"
echo "- Markdown: prettier (formatting), markdownlint (linting)"
echo "- JSON/YAML: prettier (formatting)"
echo "- General: end-of-line fixing, trailing whitespace removal"
echo ""
echo "These hooks will run automatically on every commit."
echo "To run manually: pre-commit run --all-files"
