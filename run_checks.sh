#!/bin/bash

# Spell-check
echo "Spell-checking."
codespell .
if [ $? -ne 0 ]; then
    echo "Spell-checking failed."
    exit 1
fi

# Run ruff for linting
echo "Linting with `ruff check`."
ruff check --fix --output-format concise
if [ $? -ne 0 ]; then
    echo "ruff failed."
    exit 1
fi

# Run mypy for type checking
echo "Type checking with mypy."
mypy pelicun
if [ $? -ne 0 ]; then
    echo "mypy failed. Exiting."
    exit 1
fi

# Run pytest for testing and generate coverage report
echo "Running unit-tests."
python -m pytest pelicun/tests --cov=pelicun --cov-report html -n auto
if [ $? -ne 0 ]; then
    echo "pytest failed. Exiting."
    exit 1
fi

echo "All checks passed successfully."
