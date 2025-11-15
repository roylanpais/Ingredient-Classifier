import json
import os
import pickle
import warnings
from pathlib import Path

import pandas as pd
from pycaret.classification import ClassificationExperiment
from sklearn.preprocessing import LabelEncoder

from preprocessing import TextPreprocessor

warnings.filterwarnings('ignore')

DATA_DIR = 'data'
MODEL_DIR = 'models'
OUTPUT_DIR = 'outputs'
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PyCaret configuration
PYCARET_CONFIG = {
    'normalize': True,
    'remove_outliers': True,
    'outliers_threshold': 0.05,
    'n_jobs': -1,
    'session_id': 42,
    'verbose': False,
    'remove_stopwords': True,
    'train_size': 0.7,
}


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Load training data and apply preprocessing.
    
    Args:
        filepath: Path to training CSV file.
        
    Returns:
        Preprocessed DataFrame.
        
    Raises:
        FileNotFoundError: If data file doesn't exist.
        ValueError: If data is empty or invalid.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training data not found at {filepath}")
    
    df = pd.read_csv(filepath)
    
    if df.empty:
        raise ValueError("Training dataset is empty")
    
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("DataFrame must contain 'text' and 'label' columns")
    
    print(f"✓ Loaded {len(df)} training samples from {filepath}")
    
    preprocessor = TextPreprocessor(remove_stopwords=PYCARET_CONFIG['remove_stopwords'])
    
    df['text'] = df['text'].apply(lambda x: preprocessor.preprocess(str(x)))
    
    print(f"✓ Preprocessing completed")
    print(f"  Classes: {df['label'].unique()}")
    print(f"  Class distribution:\n{df['label'].value_counts()}\n")
    
    return df


def train_model(df: pd.DataFrame) -> tuple:
    """
    Train and select best model using PyCaret.
    
    Args:
        df: DataFrame with 'text' and 'label' columns.
        
    Returns:
        Tuple of (best_model, label_encoder, exp).
    """
    print("=" * 60)
    print("MODEL TRAINING WITH PYCARET")
    print("=" * 60)
    
    exp = ClassificationExperiment()
    
    try:
        exp.setup(
            data=df,
            target='label',
            text_features=['text'],
            train_size=PYCARET_CONFIG['train_size'],
            fold=10,
            session_id =PYCARET_CONFIG['session_id'],
            verbose=PYCARET_CONFIG['verbose'],
            normalize=PYCARET_CONFIG['normalize'],
            n_jobs=PYCARET_CONFIG['n_jobs'],
        )
        print("✓ PyCaret setup completed\n")
    except Exception as e:
        print(f"✗ PyCaret setup failed: {str(e)}")
        raise
    
    # Compare models
    print("Testing multiple models...")
    try:
        top3 = exp.compare_models(n_select = 3, sort = 'F1')
        tuned_top3 = [exp.tune_model(i, optimize = 'F1') for i in top3]
        blender = exp.blend_models(tuned_top3)
        stacker = exp.stack_models(tuned_top3)
        best_model = exp.automl(optimize = 'F1')
        print("\n✓ Model comparison completed")
        
        # Pull comparison results
        model_comparison = exp.pull()
        model_comparison.to_csv(os.path.join(OUTPUT_DIR, 'models_comparison.csv'), index=False)
        print(f"  Model comparison saved to {OUTPUT_DIR}/models_comparison.csv")
        
        print(f"\n✓ Best model: {type(best_model).__name__}")
    except Exception as e:
        print(f"✗ Model comparison failed: {str(e)}")
        raise
    
    # Get label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(df['label'])
    
    print(f"\n✓ Label encoder created with classes: {list(label_encoder.classes_)}")
    
    return best_model, label_encoder, exp


def save_artifacts(exp, best_model, label_encoder: LabelEncoder) -> dict:
    """
    Save trained model and label encoder.
    
    Args:
        best_model: Trained PyCaret model.
        label_encoder: LabelEncoder for target classes.
        
    Returns:
        Dictionary with paths of saved artifacts.
    """
    print("\n" + "=" * 60)
    print("SAVING MODEL ARTIFACTS")
    print("=" * 60)
    
    model_path = os.path.join(MODEL_DIR, 'best_model')
    exp.save_model(best_model, model_path)

    encoder_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✓ Label encoder saved: {encoder_path}")
    
    return {
        'model_path': model_path,
        'encoder_path': encoder_path,
    }


def get_classification_metrics(exp, best_model) -> dict:
    """
    Extract classification metrics from PyCaret experiment.
    
    Args:
        exp: PyCaret ClassificationExperiment object.
        best_model: Trained model.
        
    Returns:
        Dictionary of metrics.
    """
    metrics = exp.pull()
    metrics.reset_index(drop=True, inplace=True)
    metrics_dict = {
        'accuracy': float(metrics.loc[0, 'Accuracy']) if 'Accuracy' in metrics.columns else None,
        'precision': float(metrics.loc[0, 'Prec.']) if 'Prec.' in metrics.columns else None,
        'recall': float(metrics.loc[0, 'Recall']) if 'Recall' in metrics.columns else None,
        'f1': float(metrics.loc[0, 'F1']) if 'F1' in metrics.columns else None,
        'auc': float(metrics.loc[0, 'AUC']) if 'AUC' in metrics.columns else None,
        'model_name': type(best_model).__name__,
    }
    
    return metrics_dict


def main():
    """Main training pipeline."""
    try:
        df_train = load_and_preprocess_data(TRAIN_FILE)
        
        best_model, label_encoder, exp = train_model(df_train)
        artifacts = save_artifacts(exp, best_model, label_encoder)

        metrics = get_classification_metrics(exp, best_model)
        metrics_path = os.path.join(OUTPUT_DIR, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Metrics saved: {metrics_path}")
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Model: {artifacts['model_path']}")
        print(f"Encoder: {artifacts['encoder_path']}")
        print(f"Metrics: {metrics_path}")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
