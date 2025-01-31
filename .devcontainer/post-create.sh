#!/bin/bash

# Install Node dependencies
npm install --save-dev

# Install Python dependencies
uv sync --all-groups --link-mode=copy

# Install pre-commit hooks
uv run pre-commit install --install-hooks