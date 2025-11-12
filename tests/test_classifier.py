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
    assert get_prediction(model, "ground cumin") == "ingredient_only" # Added multi-word

def test_ingredient_with_qty(model):
    """Test ingredients with quantities and units."""
    assert get_prediction(model, "1 egg") == "ingredient_with_qty"
    assert get_prediction(model, "200ml milk") == "ingredient_with_qty"
    assert get_prediction(model, "1/4 cup sugar") == "ingredient_with_qty"
    assert get_prediction(model, "2-3 cloves garlic") == "ingredient_with_qty"
    assert get_prediction(model, "one large onion") == "ingredient_with_qty" # Added text-based quantity
    assert get_prediction(model, "sugar (100g)") == "ingredient_with_qty" # Added parenthesis

def test_instruction_like(model):
    """Test command-like instructions."""
    assert get_prediction(model, "mix well") == "instruction_like"
    assert get_prediction(model, "Dice the carrots") == "instruction_like"
    assert get_prediction(model, "Preheat oven") == "instruction_like"
    assert get_prediction(model, "serve immediately") == "instruction_like" # Added new case
    assert get_prediction(model, "let cool for 5 minutes") == "instruction_like" # Added new case

def test_non_food(model):
    """Test non-food items or kitchen equipment."""
    assert get_prediction(model, "wooden spoon") == "non_food"
    assert get_prediction(model, "baking tray") == "non_food"
    assert get_prediction(model, "Saran wrap") == "non_food"
    assert get_prediction(model, "a small bowl") == "non_food" # Added new case
    assert get_prediction(model, "kitchen scale") == "non_food" # Added new case

def test_ambiguous_cases(model):
    """
    Test ambiguous cases. The model's answer is what we test against,
    establishing a baseline for its behavior.
    """
    # This is tricky. Is it an instruction or an ingredient?
    # Our mock model likely learned it as an instruction.
    assert get_prediction(model, "1-inch cubes") == "instruction_like"
    
    # Just a number. Could be non-food or qty.
    pred_num = get_prediction(model, "5")
    assert pred_num in ["ingredient_with_qty", "non_food"]

    # Ambiguous noun
    pred_salt = get_prediction(model, "salt")
    assert pred_salt == "ingredient_only" # should be clear
    
    # Ambiguous phrase
    pred_serve = get_prediction(model, "for serving")
    assert pred_serve in ["instruction_like", "non_food"]


def test_empty_string(model):
    """Test how the model handles an empty string."""
    # An empty string should probably not be an ingredient or instruction.
    assert get_prediction(model, "") == "non_food"
    assert get_prediction(model, "   ") == "non_food" # Added whitespace test

# --- New Test Functions ---

def test_punctuation_cases(model):
    """Test strings with punctuation."""
    # PyCaret's default NLP preprocessor should handle punctuation.
    assert get_prediction(model, "onions, chopped") == "instruction_like"
    assert get_prediction(model, "flour.") == "ingredient_only"
    assert get_prediction(model, "(optional) 1 tsp vanilla") == "ingredient_with_qty"

def test_case_insensitivity(model):
    """Test if the model is case-insensitive (it should be)."""
    assert get_prediction(model, "MILK 200 ML") == "ingredient_with_qty"
    assert get_prediction(model, "sTiR wElL") == "instruction_like"
    assert get_prediction(model, "PLASTIC WRAP") == "non_food"

def test_stop_words(model):
    """Test strings that contain only stop words."""
    # PyCaret's default setup removes stop words.
    # An empty string after preprocessing will likely be 'non_food'.
    assert get_prediction(model, "a and the") == "non_food"
    assert get_prediction(model, "it") == "non_food"

def test_long_instructions(model):
    """Test a longer, more complex instruction."""
    long_instruction = "Place the chicken in a large bowl and pour the marinade over it."
    assert get_prediction(model, long_instruction) == "instruction_like"
