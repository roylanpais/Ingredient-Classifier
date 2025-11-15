"""
Text preprocessing and feature engineering module.

Provides utilities for cleaning, tokenizing, and extracting features from ingredient text.
"""

import re
import string
from typing import List, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextPreprocessor:
    """Handles text cleaning and preprocessing for ingredient classification."""
    
    # Common verbs associated with cooking instructions
    INSTRUCTION_VERBS = {
        'chop', 'dice', 'slice', 'cut', 'mince', 'blend', 'mix', 'stir', 'whisk',
        'fold', 'heat', 'warm', 'cook', 'bake', 'fry', 'grill', 'roast', 'boil',
        'simmer', 'sauté', 'steam', 'poach', 'braise', 'knead', 'roll', 'spread',
        'pour', 'drain', 'rinse', 'wash', 'season', 'salt', 'pepper', 'add',
        'place', 'put', 'remove', 'transfer', 'sprinkle', 'drizzle', 'layer'
    }
    
    def __init__(self, remove_stopwords: bool = True, use_lemmatization: bool = False):
        """
        Initialize the preprocessor.
        
        Args:
            remove_stopwords: Whether to remove English stopwords.
            use_lemmatization: Whether to apply lemmatization (not implemented).
        """
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    
    def clean_text(self, text: str) -> str:
        """
        Clean text: lowercase, remove punctuation, remove extra whitespace.
        
        Args:
            text: Raw text input.
            
        Returns:
            Cleaned text.
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Cleaned text.
            
        Returns:
            List of tokens.
        """
        tokens = word_tokenize(text)
        
        # Remove stopwords if configured
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        return tokens
    
    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline: clean → tokenize → rejoin.
        
        Args:
            text: Raw text input.
            
        Returns:
            Preprocessed text.
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        return ' '.join(tokens)
    
    def extract_features(self, text: str) -> dict:
        """
        Extract handcrafted features from text.
        
        Features:
        - char_count: Number of characters
        - word_count: Number of words
        - has_digits: Whether text contains digits (indicator of quantity)
        - has_verb: Whether text contains cooking verbs (indicator of instruction)
        - avg_word_length: Average word length
        
        Args:
            text: Raw text input.
            
        Returns:
            Dictionary of features.
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Basic counts
        char_count = len(text)
        words = text.lower().split()
        word_count = len(words)
        
        # Digit detection
        has_digits = bool(re.search(r'\d', text))
        
        # Verb detection
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        has_verb = any(token in self.INSTRUCTION_VERBS for token in tokens)
        
        # Average word length
        avg_word_length = char_count / word_count if word_count > 0 else 0
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'has_digits': int(has_digits),
            'has_verb': int(has_verb),
            'avg_word_length': avg_word_length
        }


def validate_data(text: str) -> bool:
    """
    Validate text data for basic quality checks.
    
    Args:
        text: Input text to validate.
        
    Returns:
        True if valid, False otherwise.
    """
    if not isinstance(text, str):
        return False
    
    if len(text.strip()) == 0:
        return False
    
    return True
