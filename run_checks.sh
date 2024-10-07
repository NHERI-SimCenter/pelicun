#!/bin/bash

# # Run ruff for linting
# ruff check pelicun
# if [ $? -ne 0 ]; then
#     echo "ruff failed."
# fi

# Run flake8 for linting
flake8 pelicun
if [ $? -ne 0 ]; then
    echo "flake8 failed."
fi

# # Run pylint for additional linting
# pylint -j0 pelicun
# if [ $? -ne 0 ]; then
#     echo "pylint failed. Exiting."
#     exit 1
# fi

# # Run mypy for type checking
# mypy pelicun --no-namespace-packages
# if [ $? -ne 0 ]; then
#     echo "mypy failed. Exiting."
#     exit 1
# fi

# Run pytest for testing and generate coverage report
python -m pytest pelicun/tests --cov=pelicun --cov-report html
if [ $? -ne 0 ]; then
    echo "pytest failed. Exiting."
    exit 1
fi

echo "All checks passed successfully."
