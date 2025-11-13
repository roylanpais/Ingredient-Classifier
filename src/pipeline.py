import pandas as pd
import json
import logging
from pycaret.classification import (
    setup,
    compare_models,
    pull,
    finalize_model,
    save_model,
    load_model,
    predict_model
)
from src import config
from src import data_loader

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def train_model():
    """
    Trains, compares, and saves the best classification model.

    1. Loads training data.
    2. Sets up the PyCaret classification environment, specifying the text feature.
    3. Compares a list of fast, effective models, sorting by Macro F1.
    4. Saves the performance metrics of all models to metrics.json.
    5. Finalizes the best model (trains on full dataset).
    6. Saves the finalized model pipeline as a .pkl file.
    """
    log.info("Starting model training pipeline...")
    
    # 1. Load data
    df_train = data_loader.load_train_data(config.TRAIN_FILE)

    # 2. Setup PyCaret
    log.info("Setting up PyCaret classification environment...")
    s = setup(
        data=df_train,
        target=config.TARGET_COLUMN,
        text_features=[config.TEXT_COLUMN],  # Key for NLP classification
        session_id=123,
        log_experiment=False,  # We are logging manually
        verbose=False, # Reduce console noise
        preprocess=True,
        fix_imbalance=True # Good for small, potentially imbalanced datasets
    )

    # 3. Compare models
    log.info(f"Comparing models: {config.MODELS_TO_COMPARE}")
    best_model = compare_models(
        include=config.MODELS_TO_COMPARE,
        sort='F1'  # Sorts by Macro F1 for multiclass
    )

    # 4. Save metrics
    log.info("Saving model comparison metrics...")
    metrics_df = pull()
    # Convert metrics to a more friendly JSON format
    metrics_json = metrics_df.to_dict(orient='index')
    
    with open(config.METRICS_FILE, 'w') as f:
        json.dump(metrics_json, f, indent=4)
    log.info(f"Metrics saved to {config.METRICS_FILE}")
    print("\n--- Model Performance (on 30% hold-out set) ---")
    print(metrics_df)
    print("--------------------------------------------------")

    # 5. Finalize model
    log.info("Finalizing the best model...")
    final_model = finalize_model(best_model)

    # 6. Save model
    save_model(final_model, config.MODEL_PATH)
    log.info(f"Model pipeline saved successfully to {config.MODEL_PATH}.pkl")

def generate_predictions():
    """
    Generates predictions on the test.csv data.

    1. Loads the test data.
    2. Drops the empty 'label' column if it exists.
    3. Loads the trained model pipeline from disk.
    4. Generates predictions.
    5. Formats the output to (text, pred) and saves to predictions.csv.
    """
    log.info("Starting prediction pipeline...")
    
    # 1. Load test data
    df_test = data_loader.load_test_data(config.TEST_FILE)

    # 2. Drop empty label column if it exists
    #    predict_model works best on data without the target column
    if config.TARGET_COLUMN in df_test.columns:
        df_test = df_test.drop(columns=[config.TARGET_COLUMN])

    # 3. Load trained model
    log.info(f"Loading model from {config.MODEL_PATH}.pkl")
    try:
        model = load_model(config.MODEL_PATH)
    except FileNotFoundError:
        log.error(f"Model file not found at {config.MODEL_PATH}.pkl")
        log.error("Please run the 'train' command first: python src/main.py train")
        return
    except Exception as e:
        log.error(f"Error loading model: {e}")
        return

    # 4. Generate predictions
    log.info("Generating predictions on test data...")
    predictions = predict_model(model, data=df_test, verbose=False)

    # 5. Format and save output
    # The 'prediction_label' column is added by predict_model
    output_df = predictions[[config.TEXT_COLUMN, 'prediction_label']]
    output_df = output_df.rename(columns={'prediction_label': 'pred'})
    
    output_df.to_csv(config.PREDICTION_FILE, index=False)
    log.info(f"Predictions saved successfully to {config.PREDICTION_FILE}")
    print(f"\n--- Sample Predictions ({config.PREDICTION_FILE}) ---")
    print(output_df.head())
    print("--------------------------------------------------")