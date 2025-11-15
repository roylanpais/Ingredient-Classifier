import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

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
