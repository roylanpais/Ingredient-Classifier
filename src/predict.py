import json
import os
import pickle
import warnings
from typing import Tuple

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from preprocessing import TextPreprocessor
from pycaret.classification import load_model, predict_model

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = 'data'
MODEL_DIR = 'models'
OUTPUT_DIR = 'outputs'
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

MODEL_PATH = os.path.join(MODEL_DIR, 'best_model')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, 'predictions.csv')
TEST_METRICS_FILE = os.path.join(OUTPUT_DIR, 'test_metrics.json')


def load_model_and_encoder() -> Tuple:
    """
    Load trained model and label encoder from disk.
    
    Returns:
        Tuple of (model, label_encoder).
        
    Raises:
        FileNotFoundError: If model or encoder files don't exist.
    """
    if not os.path.exists(MODEL_PATH + ".pkl"):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Label encoder not found at {ENCODER_PATH}. Run train.py first.")
    
    model = load_model(MODEL_PATH)

    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f" Label encoder loaded: {ENCODER_PATH}")
    
    return model, label_encoder


def load_test_data(filepath: str) -> pd.DataFrame:
    """
    Load test data and apply preprocessing.
    
    Args:
        filepath: Path to test CSV file.
        
    Returns:
        Preprocessed test DataFrame.
        
    Raises:
        FileNotFoundError: If test file doesn't exist.
        ValueError: If data is empty or invalid.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Test data not found at {filepath}")
    
    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError("Test dataset is empty")
    
    if 'text' not in df.columns:
        raise ValueError("Test DataFrame must contain 'text' column")
    
    print(f" Loaded {len(df)} test samples from {filepath}")
    preprocessor = TextPreprocessor(remove_stopwords=True)
    df['text'] = df['text'].apply(lambda x: preprocessor.preprocess(str(x)))
    print(f" Test data preprocessing completed\n")
    
    return df


def make_predictions(model, label_encoder: LabelEncoder, test_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions on test data.
    
    Args:
        model: Trained model.
        label_encoder: LabelEncoder for target classes.
        test_data: Test DataFrame with 'text' column.
        
    Returns:
        DataFrame with original text and predictions.
    """
    print("=" * 60)
    print("GENERATING PREDICTIONS")
    print("=" * 60)
    try:
        predictions = predict_model(model, data = test_data[['text']])
        if isinstance(predictions, (list, pd.Series)):
            pred_labels = label_encoder.inverse_transform(predictions)
        else:
            pred_labels = predictions
    except Exception as e:
        print(f"Warning: Standard predict failed, attempting alternative: {str(e)}")
        try:
            predictions = predict_model(model, data = test_data)
            pred_labels = label_encoder.inverse_transform(predictions)
        except Exception as e2:
            raise RuntimeError(f"Failed to generate predictions: {str(e2)}")
            
    pred_labels = pred_labels[["text", "prediction_label"]]
    results = pred_labels.rename(columns = {"prediction_label": "pred"})
    
    print(f" Predictions generated: {len(results)} samples")
    print(f"  Predicted classes: {results['pred'].unique()}")
    print(f"  Class distribution:\n{results['pred'].value_counts()}\n")
    
    return results

def save_predictions(results: pd.DataFrame):
    """
    Save predictions and metrics to files.
    
    Args:
        results: DataFrame with predictions.
    """
    results.to_csv(PREDICTIONS_FILE, index=False)
    print(f" Predictions saved: {PREDICTIONS_FILE}")
    
    print(f"\nOutput file:")
    print(f"  - {PREDICTIONS_FILE}")

def main():
    """Main inference pipeline."""
    try:
        print("=" * 60)
        print("INFERENCE PIPELINE")
        print("=" * 60 + "\n")
        
        model, label_encoder = load_model_and_encoder()
        print()

        test_data = load_test_data(TEST_FILE)
        results = make_predictions(model, label_encoder, test_data)
        save_predictions(results)
        
        print("=" * 60)
        print("INFERENCE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n Inference failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
