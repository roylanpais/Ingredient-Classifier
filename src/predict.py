"""
Inference and prediction generation module.

This module handles:
- Loading trained model and label encoder
- Making predictions on test data
- Computing classification metrics
- Exporting predictions to CSV
"""

import json
import os
import pickle
import warnings
from typing import Tuple

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from preprocessing import TextPreprocessor

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = 'data'
MODEL_DIR = 'models'
OUTPUT_DIR = 'outputs'
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
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
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Label encoder not found at {ENCODER_PATH}. Run train.py first.")
    
    # Load model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded: {MODEL_PATH}")
    
    # Load encoder
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"✓ Label encoder loaded: {ENCODER_PATH}")
    
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
    
    print(f"✓ Loaded {len(df)} test samples from {filepath}")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(remove_stopwords=True)
    
    # Apply preprocessing
    df['text'] = df['text'].apply(lambda x: preprocessor.preprocess(str(x)))
    
    print(f"✓ Test data preprocessing completed\n")
    
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
    
    # Note: The actual prediction depends on model type
    # PyCaret models typically use .predict() method
    try:
        # Get predictions (assuming model.predict returns encoded labels)
        predictions = model.predict(test_data[['text']])
        
        # If predictions are numeric, decode them
        if isinstance(predictions, (list, pd.Series)):
            pred_labels = label_encoder.inverse_transform(predictions)
        else:
            # If already string labels
            pred_labels = predictions
    except Exception as e:
        print(f"Warning: Standard predict failed, attempting alternative: {str(e)}")
        # Fallback: try with the full dataframe
        try:
            predictions = model.predict(test_data)
            pred_labels = label_encoder.inverse_transform(predictions)
        except Exception as e2:
            raise RuntimeError(f"Failed to generate predictions: {str(e2)}")
    
    # Create output dataframe
    results = pd.DataFrame({
        'text': test_data['text'].values,
        'pred': pred_labels
    })
    
    print(f"✓ Predictions generated: {len(results)} samples")
    print(f"  Predicted classes: {results['pred'].unique()}")
    print(f"  Class distribution:\n{results['pred'].value_counts()}\n")
    
    return results


def compute_metrics(y_true, y_pred, label_encoder: LabelEncoder) -> dict:
    """
    Compute classification metrics (only if true labels available).
    
    Args:
        y_true: True labels (or None).
        y_pred: Predicted labels.
        label_encoder: LabelEncoder for class names.
        
    Returns:
        Dictionary of metrics or empty dict if no true labels.
    """
    if y_true is None or len(y_true) == 0:
        print("⊘ True labels not available - skipping metrics computation")
        return {}
    
    print("=" * 60)
    print("CLASSIFICATION METRICS")
    print("=" * 60)
    
    # Encode true labels for metric computation
    y_true_encoded = label_encoder.transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)
    
    # Compute metrics
    accuracy = (y_true_encoded == y_pred_encoded).mean()
    macro_f1 = f1_score(y_true_encoded, y_pred_encoded, average='macro')
    weighted_f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted')
    
    # Per-class metrics
    report = classification_report(
        y_true_encoded, y_pred_encoded,
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true_encoded, y_pred_encoded)
    
    metrics = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class_metrics': {
            label: {
                'precision': float(report[label]['precision']),
                'recall': float(report[label]['recall']),
                'f1': float(report[label]['f1-score']),
                'support': int(report[label]['support'])
            }
            for label in label_encoder.classes_
        },
        'confusion_matrix': conf_matrix.tolist(),
    }
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print("\nPer-class metrics:")
    for label in label_encoder.classes_:
        prec = metrics['per_class_metrics'][label]['precision']
        rec = metrics['per_class_metrics'][label]['recall']
        f1 = metrics['per_class_metrics'][label]['f1']
        supp = metrics['per_class_metrics'][label]['support']
        print(f"  {label:20s} - Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}, Support: {supp}")
    
    print("=" * 60 + "\n")
    
    return metrics


def save_predictions(results: pd.DataFrame, metrics: dict = None):
    """
    Save predictions and metrics to files.
    
    Args:
        results: DataFrame with predictions.
        metrics: Dictionary of metrics or None.
    """
    # Save predictions
    results.to_csv(PREDICTIONS_FILE, index=False)
    print(f"✓ Predictions saved: {PREDICTIONS_FILE}")
    
    # Save metrics if available
    if metrics:
        with open(TEST_METRICS_FILE, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Metrics saved: {TEST_METRICS_FILE}")
    
    print(f"\nOutput files:")
    print(f"  - {PREDICTIONS_FILE}")
    if metrics:
        print(f"  - {TEST_METRICS_FILE}")


def main():
    """Main inference pipeline."""
    try:
        print("=" * 60)
        print("INFERENCE PIPELINE")
        print("=" * 60 + "\n")
        
        # Load model and encoder
        model, label_encoder = load_model_and_encoder()
        print()
        
        # Load test data
        test_data = load_test_data(TEST_FILE)
        
        # Generate predictions
        results = make_predictions(model, label_encoder, test_data)
        
        # Try to compute metrics if true labels exist
        metrics = {}
        if 'label' in pd.read_csv(TEST_FILE).columns:
            test_data_original = pd.read_csv(TEST_FILE)
            # Filter out empty labels
            mask = test_data_original['label'].notna() & (test_data_original['label'] != '')
            if mask.any():
                y_true = test_data_original.loc[mask, 'label'].values
                y_pred = results.loc[mask, 'pred'].values
                metrics = compute_metrics(y_true, y_pred, label_encoder)
        
        # Save outputs
        save_predictions(results, metrics if metrics else None)
        
        print("=" * 60)
        print("INFERENCE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Inference failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
