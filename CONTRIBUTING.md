# Contributing to unnet

Thank you for your interest in contributing to unnet! This document provides guidelines and instructions for setting up your development environment and contributing to the project.

## Development Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/unnet.git
cd unnet
```

2. Install all dependencies (including dev dependencies):
```bash
uv sync --all-extras
```

This will create a virtual environment and install all required packages including development tools.

## Running Tests

Run all tests:
```bash
uv run pytest
```

Run a specific test file:
```bash
uv run pytest tests/test_grad.py
```

Run a specific test by name:
```bash
uv run pytest -k test_name
```

Run tests with verbose output:
```bash
uv run pytest -v
```

## Pre-commit Hooks

We use pre-commit hooks to automatically check code quality before commits.

### Install pre-commit hooks:
```bash
uv run pre-commit install
```

### Run pre-commit on all files manually:
```bash
uv run pre-commit run --all-files
```

The pre-commit hooks will automatically run on every commit and check:
- Ruff linting (with auto-fix)
- Ruff formatting
- Trailing whitespace

## Project Structure

```
unnet/
├── unnet/              # Main package source code
│   ├── grad.py        # Computational graph and automatic differentiation
│   ├── nn.py          # Neural network components (Neuron, Layer, Network)
│   └── utils.py       # Utilities (graph visualization)
├── tests/             # Test suite
├── notebooks/         # Jupyter notebooks with examples
│   ├── grad.ipynb    # Forward/backward propagation examples
│   └── nn.ipynb      # Neural network training examples
└── pyproject.toml    # Project configuration and dependencies
```

## Questions or Issues?

If you have questions or run into issues, please open an issue on GitHub.
