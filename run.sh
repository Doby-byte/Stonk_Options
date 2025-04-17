#!/bin/bash
# =============================================================================
# S&P 500 Options Analysis Tool - Launcher Script
# =============================================================================
# This script launches the S&P 500 Options Analysis Tool by:
# 1. Activating a Python virtual environment if available
# 2. Installing required dependencies
# 3. Running the Flask application
# =============================================================================

# Activate virtual environment if it exists
# First check if venv is in parent directory, then in current directory
if [ -d "../venv" ]; then
    source ../venv/bin/activate
    echo "Activated virtual environment from parent directory"
elif [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated virtual environment from current directory"
else
    echo "No virtual environment found, using system Python"
fi

# Install requirements if needed
echo "Installing/updating required packages..."
pip3 install -r requirements.txt

# Run the application
echo "Starting S&P 500 Options Analysis Tool..."
python3 app.py 