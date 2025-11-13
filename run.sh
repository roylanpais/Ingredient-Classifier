#!/bin/bash
set -e

# --- Configuration ---
PROJECT_DIR="ingredient-classifier"
VENV_NAME="venv"
PARENT_DIR=$(dirname "$PWD")
TRAIN_FILE_SRC="$PARENT_DIR/train.csv"
TEST_FILE_SRC="$PARENT_DIR/test.csv"

echo "--- Starting Ingredient Classifier Pipeline ---"
echo "Current directory: $(pwd)"

# 1. Create Python Virtual Environment
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment at $VENV_NAME..."
    python3 -m venv $VENV_NAME
else
    echo "Virtual environment $VENV_NAME already exists."
fi

# 2. Activate Virtual Environment
echo "Activating virtual environment..."
source $VENV_NAME/bin/activate

# 3. Install Dependencies
echo "Installing requirements from requirements.txt..."
pip install -q -r requirements.txt

# 4. Create required directories
echo "Ensuring data, models, and outputs directories exist..."
mkdir -p data
mkdir -p models
mkdir -p outputs

# 5. Copy data files (assuming they are in the parent dir)
if [ -f "$TRAIN_FILE_SRC" ]; then
    echo "Copying train.csv to data/..."
    cp "$TRAIN_FILE_SRC" data/train.csv
else
    echo "WARNING: train.csv not found at $TRAIN_FILE_SRC. Please place it in the data/ directory."
fi

if [ -f "$TEST_FILE_SRC" ]; then
    echo "Copying test.csv to data/..."
    cp "$TEST_FILE_SRC" data/test.csv
else
    echo "WARNING: test.csv not found at $TEST_FILE_SRC. Please place it in the data/ directory."
fi

# 6. Run Tests
echo "--- Running Tests ---"
pytest
echo "--- Tests Passed ---"

# 7. Run Training Pipeline
echo "--- Running Training Pipeline ---"
python3 src/main.py train
echo "--- Training Complete ---"

# 8. Run Prediction Pipeline
echo "--- Running Prediction Pipeline ---"
python3 src/main.py predict
echo "--- Prediction Complete ---"

echo ""
echo "✅ Pipeline finished successfully!"
echo "Check 'outputs/' for 'predictions.csv' and 'metrics.json'."
deactivate