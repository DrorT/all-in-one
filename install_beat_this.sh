#!/bin/bash
# Installation script for beat_this library
# This script installs beat_this and its dependencies

set -e  # Exit on error

echo "========================================"
echo "Installing beat_this for Beat Comparison"
echo "========================================"
echo ""

# Detect Python
if [ -n "$1" ]; then
    PYTHON="$1"
else
    # Default to venv Python if available
    if [ -f ~/venvs/pydemucs/bin/python ]; then
        PYTHON=~/venvs/pydemucs/bin/python
    else
        PYTHON=python3
    fi
fi

echo "Using Python: $PYTHON"
echo ""

# Check PyTorch version
echo "Checking PyTorch version..."
PYTORCH_VERSION=$($PYTHON -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not_installed")

if [ "$PYTORCH_VERSION" = "not_installed" ]; then
    echo "ERROR: PyTorch is not installed!"
    echo "Please install PyTorch 2.0 or later from: https://pytorch.org/"
    exit 1
fi

echo "PyTorch version: $PYTORCH_VERSION"

# Check if version is 2.0 or later
MAJOR_VERSION=$($PYTHON -c "import torch; print(torch.__version__.split('.')[0])")
if [ "$MAJOR_VERSION" -lt 2 ]; then
    echo "WARNING: PyTorch 2.0+ is required, you have $PYTORCH_VERSION"
    echo "Please upgrade PyTorch from: https://pytorch.org/"
    exit 1
fi

echo "✓ PyTorch version OK"
echo ""

# Install dependencies
echo "Installing dependencies..."
$PYTHON -m pip install tqdm einops soxr rotary-embedding-torch
echo "✓ Dependencies installed"
echo ""

# Install beat_this
echo "Installing beat_this from GitHub..."
$PYTHON -m pip install https://github.com/CPJKU/beat_this/archive/main.zip
echo "✓ beat_this installed"
echo ""

# Check installation
echo "Verifying installation..."
$PYTHON -c "from beat_this.inference import File2Beats; print('✓ beat_this import successful')"
echo ""

# Check ffmpeg
echo "Checking for ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo "✓ ffmpeg is installed"
else
    echo "⚠ WARNING: ffmpeg is not installed"
    echo "  For audio format support beyond .wav, install ffmpeg:"
    echo "  - Using conda: conda install ffmpeg"
    echo "  - Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  - macOS: brew install ffmpeg"
fi
echo ""

echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "You can now run the comparison tests:"
echo "  $PYTHON example_beat_comparison.py"
echo "  $PYTHON test_beat_comparison.py <audio_file>"
echo ""
