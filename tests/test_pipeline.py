import pytest
import pandas as pd
from src import pipeline, config, data_loader

def test_train_model(setup_test_environment):
    """
    Tests the train_model function.
    Asserts that the model .pkl file and metrics.json file are created.
    """
    # Act
    pipeline.train_model()

    # Assert
    model_file = config.MODEL_PATH.with_suffix(".pkl")
    assert model_file.exists(), "Model .pkl file was not created"
    assert config.METRICS_FILE.exists(), "metrics.json file was not created"
    
    # Check metrics file content
    import json
    with open(config.METRICS_FILE, 'r') as f:
        metrics = json.load(f)
    assert "Logistic Regression" in metrics, "Metrics for 'Logistic Regression' not found"
    assert "F1" in metrics["Logistic Regression"], "F1 score not found in metrics"

def test_generate_predictions(setup_test_environment):
    """
    Tests the generate_predictions function.
    Depends on a model being trained first.
    Asserts that predictions.csv is created with the correct format.
    """
    # Arrange: Train a model first
    pipeline.train_model()
    
    # Act: Generate predictions
    pipeline.generate_predictions()

    # Assert
    assert config.PREDICTION_FILE.exists(), "predictions.csv file was not created"
    
    # Check predictions file content
    df_preds = pd.read_csv(config.PREDICTION_FILE)
    df_test = data_loader.load_test_data(config.TEST_FILE)
    
    assert len(df_preds) == len(df_test), "Number of predictions does not match test set"
    assert "text" in df_preds.columns, "'text' column missing from predictions"
    assert "pred" in df_preds.columns, "'pred' column missing from predictions"
    
    # Check that predictions are all valid classes
    assert all(df_preds['pred'].isin(config.CLASSES)), "Predictions contain invalid classes"