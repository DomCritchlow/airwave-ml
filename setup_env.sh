#!/bin/bash
# Setup script for airwave-ml
# Creates a virtual environment and installs dependencies

set -e

VENV_NAME="${1:-.venv}"

echo "========================================"
echo "airwave-ml - Environment Setup"
echo "========================================"
echo

# Check Python version
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo "Found Python: $PYTHON_VERSION"

# Create virtual environment
echo
echo "Creating virtual environment in '$VENV_NAME'..."
$PYTHON_CMD -m venv "$VENV_NAME"

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo
echo "Installing dependencies..."
pip install -r requirements.txt

echo
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo
echo "To activate the environment, run:"
echo "  source $VENV_NAME/bin/activate"
echo
echo "To verify installation:"
echo "  python test_setup.py"
echo
echo "To generate training data:"
echo "  python scripts/generate_morse_data.py --output_dir data/synthetic/morse_v2 --num_samples 2000"
echo
echo "To train (CTC model):"
echo "  cd models/ctc && python train.py --config config.yaml"
echo
