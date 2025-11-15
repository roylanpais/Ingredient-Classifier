# Ingredient Classification NLP Project

A production-ready text classification pipeline for categorizing ingredient lines into four classes: `ingredient_only`, `ingredient_with_qty`, `instruction_like`, and `non_food`.

## Project Structure

```
ingredient-classifier/
├── data/
│   ├── train.csv           # Training dataset
│   └── test.csv            # Test dataset
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Data cleaning and feature engineering
│   ├── train.py           # Model training pipeline with PyCaret
│   └── predict.py         # Inference on test data
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_predict.py
├── models/                 # Trained model artifacts
├── outputs/                # Predictions and metrics
├── requirements.txt        # Python dependencies
├── setup.sh               # Reproducible setup script
├── README.md              # This file
└── DECISIONS.md           # Design decisions and trade-offs
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip and virtualenv

### Quick Start

1. **Clone/Navigate to project directory**
   ```bash
   cd ingredient-classifier
   ```

2. **Run setup script** (Linux/macOS)
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   **Or manual setup** (Windows/alternative)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

3. **Train the model**
   ```bash
   python src/train.py
   ```
   This will:
   - Load and preprocess training data
   - Test multiple ML models using PyCaret
   - Save the best model to `models/`
   - Output classification metrics to `outputs/metrics.json`

4. **Generate predictions on test data**
   ```bash
   python src/predict.py
   ```
   Output: `outputs/predictions.csv` with columns `text` and `pred`

5. **Run unit tests**
   ```bash
   pytest tests/ -v
   ```

## Data Format

### Training Data (`data/train.csv`)
| text | label |
|------|-------|
| Tomato | ingredient_only |
| Milk 200 ml | ingredient_with_qty |
| Chop the onions | instruction_like |
| Plastic wrap | non_food |

### Test Data (`data/test.csv`)
| text |
|------|
| Garlic |
| Sugar 20 g |
| Warm in a pan |

### Predictions Output (`outputs/predictions.csv`)
| text | pred |
|------|------|
| Garlic | ingredient_only |
| Sugar 20 g | ingredient_with_qty |

## Features & Approach

### Text Preprocessing
- **Lowercasing**: Normalize text case
- **Punctuation removal**: Clean special characters
- **Stop word removal**: Remove common words (optional)
- **Tokenization**: Split text into tokens

### Feature Engineering
- **Text length**: Number of characters and words
- **Contains digits**: Presence of numeric values (indicates quantities)
- **Verb presence**: Heuristic for instruction detection
- **TF-IDF vectorization**: Convert text to numerical features

### Model Selection (PyCaret)
- **Automatic testing**: PyCaret evaluates multiple algorithms
- **Best model selection**: Based on cross-validation performance
- **Algorithms tested**: Logistic Regression, Random Forest, Gradient Boosting, SVM, Naive Bayes, etc.

### Evaluation Metrics
- **Macro F1 Score**: Unweighted mean F1 across all classes
- **Precision, Recall**: Per-class performance
- **Confusion Matrix**: Error analysis

## Configuration

Edit `src/train.py` to customize:

```python
PYCARET_CONFIG = {
    'normalize': True,
    'remove_stopwords': True,
    'remove_outliers': True,
    'outliers_threshold': 0.05,
    'n_jobs': -1  # Parallel processing
}
```

## Outputs

After running `python src/train.py`:
- `models/best_model.pkl` - Serialized trained model
- `models/label_encoder.pkl` - Label encoding for class names
- `outputs/metrics.json` - Classification metrics
- `outputs/models_comparison.csv` - PyCaret model comparison

After running `python src/predict.py`:
- `outputs/predictions.csv` - Test predictions
- `outputs/test_metrics.json` - Test set evaluation metrics (if labels available)

## Testing

The project includes comprehensive unit tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

Tests cover:
- Data preprocessing edge cases (empty strings, special characters, length extremes)
- Text feature extraction
- Model training pipeline
- Inference on unseen data

## Performance

**Classification Metrics on Test Set**
- Classes: ingredient_only, ingredient_with_qty, instruction_like, non_food
- Macro F1 Score: Computed during evaluation
- See `outputs/metrics.json` for detailed metrics

## Troubleshooting

| Issue | Solution |
|-------|----------|
| PyCaret GPU error | Set `gpu_param=False` in training config |
| NLTK resource error | Run: `python -c "import nltk; nltk.download('punkt')"` |
| Memory issues | Reduce `n_jobs` parameter or filter large datasets |
| Import errors | Verify all packages: `pip install -r requirements.txt` |

## Dependencies

See `requirements.txt` for all dependencies. Key libraries:
- **pandas**: Data manipulation
- **scikit-learn**: ML algorithms and metrics
- **pycaret**: Low-code ML automation
- **pytest**: Unit testing framework

## License

Internal project. All rights reserved.

## Contact

For issues or questions, refer to `DECISIONS.md` for architecture details.
