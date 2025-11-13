import pytest
import pandas as pd
from pycaret.classification import load_model, predict_model
from src import config

@pytest.fixture(scope="module")
def trained_model(setup_test_environment):
    """
    Fixture that re-uses the test environment setup to train
    and return a loaded model object for testing.
    """
    from src import pipeline
    # setup_test_environment has already run and patched config
    pipeline.train_model()
    model = load_model(config.MODEL_PATH)
    return model

def test_prediction_edge_cases(trained_model):
    """
    Tests the trained model's robustness against edge-case inputs:
    - Empty string
    - Numbers only
    - Punctuation only
    - A mix of unseen, but reasonable, inputs
    
    Asserts that the model does not crash and returns a valid class label.
    """
    # Arrange
    edge_cases = {
        config.TEXT_COLUMN: [
            "",                  # Empty string
            "12345",             # Numbers only
            "!!!",               # Punctuation only
            "A single onion",    # Unseen 'ingredient_only' style
            "3 cups of flour",   # Unseen 'ingredient_with_qty' style
            "Whisk vigorously",  # Unseen 'instruction_like' style
            "Parchment paper"    # Unseen 'non_food' (from train.csv)
        ]
    }
    df_edges = pd.DataFrame(edge_cases)

    # Act
    predictions = predict_model(trained_model, data=df_edges, verbose=False)

    # Assert
    assert len(predictions) == len(df_edges), "Prediction count mismatch"
    assert 'prediction_label' in predictions.columns, "prediction_label column missing"
    
    # Main assertion: All predictions must be one of the known classes
    pred_labels = predictions['prediction_label']
    assert all(pred_labels.isin(config.CLASSES)), "Model produced an invalid class label"

    # Specific check for empty string handling
    empty_pred = predictions[predictions[config.TEXT_COLUMN] == '']['prediction_label']
    assert len(empty_pred) == 1, "Empty string was not processed"
    assert empty_pred.iloc[0] in config.CLASSES, "Empty string prediction is not a valid class"