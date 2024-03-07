#!/bin/bash

# Default source directory and options for the tools
SOURCE_DIR="."
AUTOFIX_OPTIONS="--remove-all-unused-imports --remove-unused-variables --expand-star-imports --ignore-init-module-imports --in-place -r"
ISORT_OPTIONS="--profile black --line-length 88"
BLACK_OPTIONS="--line-length 88"


# Install the necessary packages
pip install autoflake isort black

# Run autoflake with the specified options
output=$(autoflake $SOURCE_DIR $AUTOFIX_OPTIONS)
if [ -n "$output" ]; then
  echo "Autoflake made changes or found issues:"
  echo "$output"
  # Uncomment the next line if you want the script to fail on changes
  # exit 1
else
  echo "No issues found by autoflake."
fi

# Run isort with the specified options
echo "Running isort..."
isort $SOURCE_DIR $ISORT_OPTIONS

# Run black with the specified options
echo "Running black..."
black $SOURCE_DIR $BLACK_OPTIONS

# Final message
echo "Linting complete."
