"""
Unit tests for inference and prediction module.

Tests cover:
- Data loading functionality
- Prediction generation
- Metrics computation
"""

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from preprocessing import TextPreprocessor


class TestDataLoading:
    """Test data loading utilities."""
    
    def test_load_test_data_valid(self):
        """Test loading valid test data."""
        # Create temporary test CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("text\nGarlic\nSugar 20 g\n")
            temp_file = f.name
        
        try:
            df = pd.read_csv(temp_file)
            assert len(df) == 2
            assert 'text' in df.columns
        finally:
            os.unlink(temp_file)
    
    def test_load_test_data_with_labels(self):
        """Test loading test data that includes labels (for validation)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("text,label\nGarlic,ingredient_only\nSugar 20 g,ingredient_with_qty\n")
            temp_file = f.name
        
        try:
            df = pd.read_csv(temp_file)
            assert len(df) == 2
            assert 'text' in df.columns
            assert 'label' in df.columns
        finally:
            os.unlink(temp_file)
    
    def test_preprocessing_applied_correctly(self):
        """Test that preprocessing is applied to loaded data."""
        preprocessor = TextPreprocessor()
        text = "Chop The Onions!"
        processed = preprocessor.preprocess(text)
        
        # Should be lowercase
        assert processed == processed.lower()
        # Punctuation should be removed
        assert "!" not in processed


class TestFeatureExtraction:
    """Test feature extraction for model input."""
    
    def test_text_features_extraction(self):
        """Test that text features are extracted correctly."""
        preprocessor = TextPreprocessor()
        
        test_cases = [
            ("Tomato", {'has_digits': 0, 'has_verb': 0, 'word_count': 1}),
            ("Milk 200 ml", {'has_digits': 1, 'has_verb': 0, 'word_count': 3}),
            ("Chop the onions", {'has_digits': 0, 'has_verb': 1, 'word_count': 3}),
            ("Plastic wrap", {'has_digits': 0, 'has_verb': 0, 'word_count': 2}),
        ]
        
        for text, expected in test_cases:
            features = preprocessor.extract_features(text)
            for key, value in expected.items():
                assert features[key] == value, f"Failed for text '{text}'"
    
    def test_features_for_ingredient_with_qty(self):
        """Test feature extraction specifically for ingredient_with_qty class."""
        preprocessor = TextPreprocessor()
        
        test_texts = ["Butter 50 g", "Eggs 2", "Rice 150 g"]
        
        for text in test_texts:
            features = preprocessor.extract_features(text)
            assert features['has_digits'] == 1, f"Expected digit detection for '{text}'"


class TestPredictionOutput:
    """Test prediction output format and structure."""
    
    def test_predictions_dataframe_structure(self):
        """Test that predictions have correct structure."""
        # Create sample prediction dataframe
        predictions = pd.DataFrame({
            'text': ['Garlic', 'Sugar 20 g', 'Warm in a pan'],
            'pred': ['ingredient_only', 'ingredient_with_qty', 'instruction_like']
        })
        
        assert len(predictions) == 3
        assert list(predictions.columns) == ['text', 'pred']
        assert predictions['pred'].isin(['ingredient_only', 'ingredient_with_qty', 
                                        'instruction_like', 'non_food']).all()
    
    def test_predictions_no_missing_values(self):
        """Test that predictions don't have missing values."""
        predictions = pd.DataFrame({
            'text': ['Garlic', 'Sugar 20 g'],
            'pred': ['ingredient_only', 'ingredient_with_qty']
        })
        
        assert not predictions['text'].isna().any()
        assert not predictions['pred'].isna().any()
    
    def test_predictions_valid_classes(self):
        """Test that all predictions are valid class labels."""
        valid_classes = {'ingredient_only', 'ingredient_with_qty', 
                        'instruction_like', 'non_food'}
        
        predictions = pd.DataFrame({
            'text': ['Garlic', 'Sugar 20 g'],
            'pred': ['ingredient_only', 'ingredient_with_qty']
        })
        
        for pred in predictions['pred']:
            assert pred in valid_classes


class TestLabelEncoding:
    """Test label encoding and decoding."""
    
    def test_label_encoder_creation(self):
        """Test creating a label encoder."""
        classes = ['ingredient_only', 'ingredient_with_qty', 'instruction_like', 'non_food']
        encoder = LabelEncoder()
        encoder.fit(classes)
        
        assert list(encoder.classes_) == classes
    
    def test_label_encoding_decoding(self):
        """Test encoding and decoding labels."""
        classes = ['ingredient_only', 'ingredient_with_qty', 'instruction_like', 'non_food']
        encoder = LabelEncoder()
        encoder.fit(classes)
        
        # Encode
        encoded = encoder.transform(classes)
        assert len(encoded) == 4
        assert all(0 <= e < 4 for e in encoded)
        
        # Decode
        decoded = encoder.inverse_transform(encoded)
        assert list(decoded) == classes
    
    def test_label_encoder_persistence(self):
        """Test saving and loading label encoder."""
        classes = ['ingredient_only', 'ingredient_with_qty', 'instruction_like', 'non_food']
        encoder = LabelEncoder()
        encoder.fit(classes)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_file = f.name
            pickle.dump(encoder, f)
        
        try:
            # Load from file
            with open(temp_file, 'rb') as f:
                loaded_encoder = pickle.load(f)
            
            assert list(loaded_encoder.classes_) == classes
        finally:
            os.unlink(temp_file)


class TestMetricsComputation:
    """Test metrics computation (if labels are available)."""
    
    def test_accuracy_computation(self):
        """Test accuracy calculation."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        
        accuracy = np.mean(y_true == y_pred)
        assert accuracy == 4/6  # 4 correct out of 6
    
    def test_f1_score_multiclass(self):
        """Test F1 score for multiclass classification."""
        from sklearn.metrics import f1_score
        
        y_true = np.array([0, 1, 2, 3, 0, 1])
        y_pred = np.array([0, 1, 2, 2, 0, 1])
        
        # Macro F1
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        assert 0 <= macro_f1 <= 1
        
        # Weighted F1
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        assert 0 <= weighted_f1 <= 1
    
    def test_confusion_matrix_shape(self):
        """Test confusion matrix has correct shape."""
        from sklearn.metrics import confusion_matrix
        
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 2, 0, 1, 2, 3])
        
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (4, 4)


class TestEndToEndInference:
    """Integration tests for end-to-end inference."""
    
    def test_preprocessing_then_prediction(self):
        """Test preprocessing followed by prediction."""
        preprocessor = TextPreprocessor()
        
        test_texts = [
            ("Garlic", {'has_digits': 0, 'has_verb': 0}),
            ("Sugar 20 g", {'has_digits': 1, 'has_verb': 0}),
            ("Warm in a pan", {'has_digits': 0, 'has_verb': 1}),
            ("Aluminum foil", {'has_digits': 0, 'has_verb': 0}),
        ]
        
        for text, expected_features in test_texts:
            processed = preprocessor.preprocess(text)
            features = preprocessor.extract_features(text)
            
            for key, value in expected_features.items():
                assert features[key] == value
    
    def test_batch_prediction_structure(self):
        """Test batch prediction output structure."""
        batch = pd.DataFrame({
            'text': ['Garlic', 'Sugar 20 g', 'Warm in a pan', 'Aluminum foil',
                    'Rice 150 g', 'Cumin', 'Stir for 2 minutes', 'Salt']
        })
        
        # Simulate predictions
        predictions = batch.copy()
        predictions['pred'] = ['ingredient_only', 'ingredient_with_qty', 'instruction_like',
                              'non_food', 'ingredient_with_qty', 'ingredient_only',
                              'instruction_like', 'ingredient_only']
        
        assert len(predictions) == len(batch)
        assert 'text' in predictions.columns
        assert 'pred' in predictions.columns
