#!/bin/bash
# setup.sh - Reproducible setup script for ingredient classification project

set -e  # Exit on error

echo "==========================================="
echo "Ingredient Classifier - Setup Script"
echo "==========================================="

# Step 1: Create virtual environment
echo "Step 1: Creating Python virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

# Step 2: Upgrade pip
echo "Step 2: Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Step 3: Install dependencies
echo "Step 3: Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Step 4: Download NLTK resources
echo "Step 4: Downloading NLTK resources..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Step 5: Create necessary directories
echo "Step 5: Creating project directories..."
mkdir -p data models outputs

# Step 6: Verify installation
echo "Step 6: Verifying installation..."
python -c "import pandas; import sklearn; import pycaret; print('✓ All dependencies installed successfully!')"

echo ""
echo "==========================================="
echo "Setup complete! Run the following commands:"
echo "==========================================="
echo "source venv/bin/activate"
echo "python src/train.py"
echo "pytest tests/"
echo "==========================================="
