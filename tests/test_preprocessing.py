import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import pytest

from src.preprocessing import TextPreprocessor, validate_data


class TestTextCleaning:
    """Test text cleaning functionality."""
    
    def test_clean_text_lowercase(self):
        """Test that text is converted to lowercase."""
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("Tomato")
        assert result == "tomato"
    
    def test_clean_text_punctuation_removal(self):
        """Test that punctuation is removed."""
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("Chop, slice, and mix!")
        assert result == "chop slice and mix"
    
    def test_clean_text_extra_whitespace(self):
        """Test that extra whitespace is removed."""
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("  Milk    200    ml  ")
        assert result == "milk 200 ml"
    
    def test_clean_text_empty_string(self):
        """Test handling of empty string."""
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("")
        assert result == ""
    
    def test_clean_text_special_characters(self):
        """Test removal of special characters."""
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("Tomato@#$%^&*()")
        assert result == "tomato"
    
    def test_clean_text_unicode(self):
        """Test handling of unicode characters."""
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("Café")
        assert "caf" in result


class TestTokenization:
    """Test tokenization functionality."""
    
    def test_tokenize_simple(self):
        """Test basic tokenization."""
        preprocessor = TextPreprocessor(remove_stopwords=False)
        result = preprocessor.tokenize("tomato onion garlic")
        assert len(result) == 3
        assert "tomato" in result
    
    def test_tokenize_with_stopword_removal(self):
        """Test stopword removal."""
        preprocessor = TextPreprocessor(remove_stopwords=True)
        result = preprocessor.tokenize("chop the onions")
        # 'the' should be removed
        assert "the" not in result
        assert "chop" in result
        assert "onions" in result
    
    def test_tokenize_without_stopword_removal(self):
        """Test that stopwords are kept when disabled."""
        preprocessor = TextPreprocessor(remove_stopwords=False)
        result = preprocessor.tokenize("chop the onions")
        assert "the" in result
    
    def test_tokenize_empty_string(self):
        """Test tokenization of empty string."""
        preprocessor = TextPreprocessor()
        result = preprocessor.tokenize("")
        assert result == []
    
    def test_tokenize_single_word(self):
        """Test tokenization of single word."""
        preprocessor = TextPreprocessor()
        result = preprocessor.tokenize("tomato")
        assert len(result) == 1
        assert result[0] == "tomato"


class TestFeatureExtraction:
    """Test handcrafted feature extraction."""
    
    def test_features_ingredient_only(self):
        """Test feature extraction for ingredient-only examples."""
        preprocessor = TextPreprocessor()
        features = preprocessor.extract_features("Tomato")
        
        assert features['char_count'] == 6
        assert features['word_count'] == 1
        assert features['has_digits'] == 0
        assert features['has_verb'] == 0
        assert features['avg_word_length'] == 6
    
    def test_features_ingredient_with_qty(self):
        """Test feature extraction for ingredient-with-qty examples."""
        preprocessor = TextPreprocessor()
        features = preprocessor.extract_features("Milk 200 ml")
        
        assert features['char_count'] == 11
        assert features['word_count'] == 3
        assert features['has_digits'] == 1  # Contains '200'
        assert features['has_verb'] == 0
        assert features['avg_word_length'] > 0
    
    def test_features_instruction_like(self):
        """Test feature extraction for instruction-like examples."""
        preprocessor = TextPreprocessor()
        features = preprocessor.extract_features("Chop the onions")
        
        assert features['word_count'] == 3
        assert features['has_digits'] == 0
        assert features['has_verb'] == 1  # 'chop' is a verb
    
    def test_features_non_food(self):
        """Test feature extraction for non-food examples."""
        preprocessor = TextPreprocessor()
        features = preprocessor.extract_features("Plastic wrap")
        
        assert features['char_count'] == 12
        assert features['word_count'] == 2
        assert features['has_digits'] == 0
        assert features['has_verb'] == 0
    
    def test_features_with_multiple_verbs(self):
        """Test feature extraction with multiple verbs."""
        preprocessor = TextPreprocessor()
        features = preprocessor.extract_features("Chop and slice the vegetables")
        
        # Should detect at least one verb
        assert features['has_verb'] == 1
    
    def test_features_empty_string(self):
        """Test feature extraction on empty string."""
        preprocessor = TextPreprocessor()
        features = preprocessor.extract_features("")
        
        assert features['char_count'] == 0
        assert features['word_count'] == 0
        assert features['has_digits'] == 0
        assert features['has_verb'] == 0
        assert features['avg_word_length'] == 0
    
    def test_features_numeric_input(self):
        """Test feature extraction with numeric input."""
        preprocessor = TextPreprocessor()
        features = preprocessor.extract_features(200)
        
        assert features['has_digits'] == 1
    
    def test_features_all_digits(self):
        """Test feature extraction with all digits."""
        preprocessor = TextPreprocessor()
        features = preprocessor.extract_features("12345")
        
        assert features['has_digits'] == 1


class TestPreprocessingPipeline:
    """Test full preprocessing pipeline."""
    
    def test_preprocess_ingredient_only(self):
        """Test preprocessing of ingredient-only text."""
        preprocessor = TextPreprocessor(remove_stopwords=True)
        result = preprocessor.preprocess("Tomato")
        assert "tomato" in result.lower()
    
    def test_preprocess_with_special_chars(self):
        """Test preprocessing text with special characters."""
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess("Chop, slice, & mix!")
        # Should remove punctuation and lowercase
        assert "," not in result
        assert "!" not in result
        assert "&" not in result
    
    def test_preprocess_multiple_spaces(self):
        """Test preprocessing with multiple spaces."""
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess("  Milk    200    ml  ")
        # Should normalize whitespace
        assert "  " not in result


class TestDataValidation:
    """Test data validation utility."""
    
    def test_validate_data_valid_string(self):
        """Test validation of valid string."""
        assert validate_data("Tomato") is True
    
    def test_validate_data_empty_string(self):
        """Test validation of empty string."""
        assert validate_data("") is False
    
    def test_validate_data_whitespace_only(self):
        """Test validation of whitespace-only string."""
        assert validate_data("   ") is False
    
    def test_validate_data_non_string(self):
        """Test validation of non-string input."""
        assert validate_data(123) is False
        assert validate_data(None) is False
        assert validate_data([]) is False
    
    def test_validate_data_normal_text(self):
        """Test validation of normal text."""
        assert validate_data("Chop the onions into pieces") is True


class TestPreprocessorConfiguration:
    """Test preprocessor configuration options."""
    
    def test_preprocessor_stopword_configuration(self):
        """Test stop word removal configuration."""
        pp_with_stop = TextPreprocessor(remove_stopwords=True)
        pp_without_stop = TextPreprocessor(remove_stopwords=False)
        
        text = "the quick brown fox"
        
        tokens_with = pp_with_stop.tokenize(text)
        tokens_without = pp_without_stop.tokenize(text)
        
        # Without stopword removal should have more tokens
        assert len(tokens_without) > len(tokens_with)
    
    def test_preprocessor_consistent_behavior(self):
        """Test that preprocessor is consistent across multiple calls."""
        preprocessor = TextPreprocessor()
        text = "Chop the onions"
        
        result1 = preprocessor.preprocess(text)
        result2 = preprocessor.preprocess(text)
        
        assert result1 == result2
