import pytest
import pandas as pd
import os
from pycaret.nlp import load_model, predict_model

# --- Constants ---
MODEL_NAME = "ingredient_model"
MODEL_FILE_PATH = f"{MODEL_NAME}.pkl"

# Fixture to load the model once for all tests
@pytest.fixture(scope="module")
def model():
    """
    Loads the trained model artifact.
    This test suite assumes the model has already been trained
    (e.g., by running `python train_predict.py --mode train`).
    """
    if not os.path.exists(MODEL_FILE_PATH):
        pytest.skip(f"Model file {MODEL_FILE_PATH} not found. Run training first.")
    
    try:
        return load_model(MODEL_NAME)
    except Exception as e:
        pytest.fail(f"Failed to load model '{MODEL_NAME}'. Error: {e}")

def get_prediction(model, text):
    """Helper function to get a single prediction label."""
    test_df = pd.DataFrame([{"text": text}])
    predictions = predict_model(model, data=test_df)
    # Get the label from the first (and only) row
    return predictions["prediction_label"].iloc[0]

# --- Test Cases ---

def test_ingredient_only(model):
    """Test simple, one-word ingredients."""
    assert get_prediction(model, "pepper") == "ingredient_only"
    assert get_prediction(model, "Chicken") == "ingredient_only"
    assert get_prediction(model, "sugar") == "ingredient_only"

def test_ingredient_with_qty(model):
    """Test ingredients with quantities and units."""
    assert get_prediction(model, "1 egg") == "ingredient_with_qty"
    assert get_prediction(model, "200ml milk") == "ingredient_with_qty"
    assert get_prediction(model, "1/4 cup sugar") == "ingredient_with_qty"
    assert get_prediction(model, "2-3 cloves garlic") == "ingredient_with_qty"

def test_instruction_like(model):
    """Test command-like instructions."""
    assert get_prediction(model, "mix well") == "instruction_like"
    assert get_prediction(model, "Dice the carrots") == "instruction_like"
    assert get_prediction(model, "Preheat oven") == "instruction_like"

def test_non_food(model):
    """Test non-food items or kitchen equipment."""
    assert get_prediction(model, "wooden spoon") == "non_food"
    assert get_prediction(model, "baking tray") == "non_food"
    assert get_prediction(model, "Saran wrap") == "non_food"

def test_ambiguous_cases(model):
    """
    Test ambiguous cases. The model's answer is what we test against,
    establishing a baseline for its behaviour.
    """
    # This is tricky. Is it an instruction or an ingredient?
    # Our mock model likely learned it as an instruction.
    assert get_prediction(model, "1-inch cubes") == "instruction_like"
    
    # Just a number. Could be non-food or qty.
    pred = get_prediction(model, "5")
    assert pred in ["ingredient_with_qty", "non_food"]

def test_empty_string(model):
    """Test how the model handles an empty string."""
    # An empty string should probably not be an ingredient or instruction.
    assert get_prediction(model, "") == "non_food"
