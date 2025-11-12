import pandas as pd
import argparse
import os
from pycaret.nlp import (
    setup,
    compare_models,
    finalize_model,
    save_model,
    load_model,
    predict_model,
    add_metric,
    pull,
)
from sklearn.metrics import f1_score

# --- Constants ---
TRAIN_FILE_PATH = os.path.join("data", "train.csv")
TEST_FILE_PATH = os.path.join("data", "test.csv")
OUTPUT_FILE_PATH = "predictions.csv"
MODEL_NAME = "ingredient_model"
MODEL_FILE_PATH = f"{MODEL_NAME}.pkl"


def train_and_evaluate():
    """
    Loads training data, sets up PyCaret NLP experiment,
    compares models to find the best one based on Macro F1,
    trains the final model on all data, and saves it.
    """
    print(f"--- 1. Starting Training Mode ---")
    
    # Load data
    try:
        df = pd.read_csv(TRAIN_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Training file not found at {TRAIN_FILE_PATH}")
        return

    print(f"Loaded {len(df)} training records.")

    # Set up PyCaret NLP experiment
    print("Setting up PyCaret NLP experiment...")
    # log_experiment=False to avoid creating MLflow files
    s = setup(
        data=df,
        target="label",
        text_features=["text"],
        session_id=123,
        log_experiment=False,
    )

    # Add Macro F1 score as a key metric
    # This is crucial for multi-class classification, especially with imbalance
    add_metric("macro_f1", "Macro F1", f1_score, average="macro")

    # Compare models and select the best one
    print("Comparing models... (This may take a few minutes)")
    # We sort by our custom 'Macro F1' metric in descending order
    best_model = compare_models(sort="Macro F1", n_select=1)

    print("\n--- Cross-Validation Results ---")
    results_df = pull()
    print(results_df)
    print(f"\nBest model found: {best_model}")

    # Finalize the model
    print("Finalizing model (retraining on full dataset)...")
    final_model = finalize_model(best_model)

    # Save the model
    print(f"Saving model to {MODEL_FILE_PATH}...")
    save_model(final_model, MODEL_NAME)
    
    print("--- Training Complete ---")


def predict_on_test():
    """
    Loads the saved model, predicts on the test data,
    and saves the predictions to a CSV file.
    """
    print(f"--- 2. Starting Prediction Mode ---")
    
    # Check if model file exists
    if not os.path.exists(MODEL_FILE_PATH):
        print(f"Error: Model file '{MODEL_FILE_PATH}' not found.")
        print("Please run training first: python train_predict.py --mode train")
        return

    # Check if test data file exists
    try:
        test_df = pd.read_csv(TEST_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Test file not found at {TEST_FILE_PATH}")
        return
        
    print(f"Loaded {len(test_df)} records for prediction.")

    # Load the trained model
    print(f"Loading model from {MODEL_FILE_PATH}...")
    model = load_model(MODEL_NAME)

    # Generate predictions
    print("Generating predictions...")
    predictions = predict_model(model, data=test_df)

    # Format the output
    # The prediction is in the 'prediction_label' column
    output_df = predictions[["text", "prediction_label"]]
    output_df = output_df.rename(columns={"prediction_label": "pred"})

    # Save the output
    output_df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"Predictions saved successfully to {OUTPUT_FILE_PATH}.")
    print("--- Prediction Complete ---")


def main():
    """
    Main function to parse command-line arguments
    and run the appropriate mode (train or predict).
    """
    parser = argparse.ArgumentParser(
        description="Train or run predictions for the Ingredient Classifier."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict"],
        required=True,
        help="Set to 'train' to train a new model, or 'predict' to generate predictions.",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_and_evaluate()
    elif args.mode == "predict":
        predict_on_test()


if __name__ == "__main__":
    main()
