# Project Execution Guide

## Quick Reference: Running the Project

### 1. Setup (One-time)
```bash
chmod +x setup.sh
./setup.sh
```

### 2. Train Model
```bash
source venv/bin/activate
python src/train.py
```

### 3. Generate Predictions
```bash
python src/predict.py
```

### 4. Run Tests
```bash
pytest tests/ -v
```

---

## Project Architecture

```
┌─────────────────────────────────────────┐
│        Ingredient Classification       │
│              NLP Pipeline              │
└─────────────────────────────────────────┘

INPUT:
├── Training Data (12 samples)
│   └── text: ingredient line
│   └── label: one of 4 classes
│
├── Test Data (8 samples)
    └── text: ingredient line (no label)

PROCESSING PIPELINE:
│
├── [1] PREPROCESSING (preprocessing.py)
│   ├── Lowercase & punctuation removal
│   ├── Tokenization
│   ├── Stop word removal
│   └── Feature extraction:
│       ├── Text length metrics
│       ├── Digit detection (for qty)
│       ├── Verb detection (for instructions)
│       └── TF-IDF vectorization
│
├── [2] MODEL TRAINING (train.py)
│   ├── Data loading & validation
│   ├── PyCaret Classification Experiment
│   │   ├── Tests multiple algorithms
│   │   ├── Stratified 10-fold CV
│   │   └── Selects best performer
│   ├── Model Serialization (pickle)
│   └── Outputs:
│       ├── models/best_model.pkl
│       ├── models/label_encoder.pkl
│       └── outputs/metrics.json
│
├── [3] INFERENCE (predict.py)
│   ├── Load model & encoder
│   ├── Preprocess test data
│   ├── Generate predictions
│   ├── Compute metrics (if labels available)
│   └── Outputs:
│       ├── outputs/predictions.csv
│       └── outputs/test_metrics.json
│
└── [4] TESTING (tests/)
    ├── test_preprocessing.py
    │   ├── Text cleaning edge cases
    │   ├── Feature extraction validation
    │   └── Tokenization correctness
    │
    └── test_predict.py
        ├── Data loading
        ├── Prediction format
        ├── Metrics computation
        └── End-to-end integration

OUTPUT:
├── predictions.csv
│   ├── text: original ingredient line
│   └── pred: predicted class
│
├── test_metrics.json
│   ├── accuracy
│   ├── macro_f1
│   ├── per_class metrics
│   └── confusion_matrix
│
└── models_comparison.csv
    └── Performance of all tested algorithms
```

---

## Data Flow

### Training Phase
```
train.csv
    ↓
[Load & Validate]
    ↓
[Preprocess]
    ├── Lowercase, remove punctuation
    ├── Tokenization
    └── Extract features
    ↓
[PyCaret Setup]
    ├── TF-IDF vectorization
    ├── Feature engineering
    └── 10-fold stratified CV
    ↓
[Model Comparison]
    ├── Logistic Regression
    ├── Random Forest
    ├── Gradient Boosting
    ├── SVM
    └── ... (9+ algorithms)
    ↓
[Select Best Model]
    ↓
[Save Artifacts]
    ├── best_model.pkl
    └── label_encoder.pkl
```

### Inference Phase
```
test.csv
    ↓
[Load Test Data]
    ↓
[Preprocess]
    ├── Same pipeline as training
    ├── Lowercase, remove punctuation
    ├── Tokenization
    └── Extract features
    ↓
[Load Model & Encoder]
    ├── best_model.pkl
    └── label_encoder.pkl
    ↓
[Generate Predictions]
    ↓
[Compute Metrics (if labels available)]
    ├── Accuracy
    ├── Macro F1 Score
    ├── Per-class precision/recall
    └── Confusion matrix
    ↓
[Export Results]
    ├── predictions.csv
    └── test_metrics.json
```

---

## Key Components

### 1. TextPreprocessor (preprocessing.py)

**Methods:**
- `clean_text()` - Lowercase, punctuation removal, whitespace normalization
- `tokenize()` - Word tokenization with optional stopword removal
- `preprocess()` - Full pipeline (clean + tokenize + rejoin)
- `extract_features()` - Handcrafted features for model input

**Key Features:**
- Verb detection for instruction classification
- Digit detection for quantity classification
- Configurable stop word removal
- Handles edge cases (empty strings, unicode, special chars)

### 2. Training Pipeline (train.py)

**Workflow:**
1. Load training CSV with validation
2. Apply preprocessing to all samples
3. Initialize PyCaret ClassificationExperiment
4. Configure automated ML settings
5. Compare multiple algorithms (automatic)
6. Select best model based on CV performance
7. Save model and label encoder
8. Export metrics and model comparison

**PyCaret Advantages:**
- Automatic handling of TF-IDF vectorization
- Multiple algorithms tested in one call
- Stratified cross-validation with 10 folds
- Built-in preprocessing pipeline
- Model serialization support

### 3. Inference Pipeline (predict.py)

**Workflow:**
1. Load pretrained model from pickle
2. Load label encoder
3. Load test data without preprocessing label column
4. Apply same preprocessing as training
5. Generate predictions using loaded model
6. If true labels available: compute metrics
7. Export predictions and optional metrics

### 4. Test Suite (tests/)

**Test Coverage:**

**test_preprocessing.py (40+ tests):**
- Text cleaning: lowercase, punctuation, whitespace
- Tokenization: basic, stopword removal, edge cases
- Feature extraction: all 4 class types
- Pipeline consistency
- Data validation

**test_predict.py (20+ tests):**
- Data loading and validation
- Feature extraction correctness
- Prediction output format
- Label encoding/decoding
- Metrics computation
- End-to-end integration

---

## File Structure

```
ingredient-classifier/
│
├── data/
│   ├── train.csv              # 12 training samples with labels
│   └── test.csv               # 8 test samples (no labels)
│
├── src/
│   ├── __init__.py            # Package initialization
│   ├── preprocessing.py       # 200+ lines: Text preprocessing & features
│   ├── train.py              # 180+ lines: PyCaret training pipeline
│   └── predict.py            # 200+ lines: Inference & metrics
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py  # 350+ lines: 40+ test cases
│   └── test_predict.py        # 300+ lines: 20+ test cases
│
├── models/                    # Trained model artifacts (created after training)
│   ├── best_model.pkl         # Serialized trained model
│   └── label_encoder.pkl      # Label encoder for class names
│
├── outputs/                   # Output files (created after prediction)
│   ├── predictions.csv        # Test predictions
│   ├── test_metrics.json      # Classification metrics
│   └── models_comparison.csv  # PyCaret model comparison
│
├── requirements.txt           # Python dependencies (pinned versions)
├── setup.sh                   # Bash setup script
├── README.md                  # Complete documentation
├── DECISIONS.md              # Architecture decisions & trade-offs
└── PROJECT_GUIDE.md          # This file
```

---

## Expected Outputs

### After `python src/train.py`:

**Console Output:**
```
✓ Loaded 12 training samples from data/train.csv
✓ Preprocessing completed
  Classes: ['ingredient_only' 'ingredient_with_qty' 'instruction_like' 'non_food']
  Class distribution:
  ingredient_only         3
  ingredient_with_qty     3
  instruction_like        3
  non_food               3

✓ PyCaret setup completed

Testing multiple models...
✓ Model comparison completed
✓ Best model: RandomForestClassifier

✓ Label encoder created with classes: [...4 classes...]

SAVING MODEL ARTIFACTS
✓ Model saved: models/best_model.pkl
✓ Label encoder saved: models/label_encoder.pkl
✓ Metrics saved: outputs/metrics.json

TRAINING COMPLETED SUCCESSFULLY
```

**Generated Files:**
- `models/best_model.pkl` - Trained model (~5-50 MB depending on algorithm)
- `models/label_encoder.pkl` - Label encoder (~1 KB)
- `outputs/metrics.json` - Training metrics
- `outputs/models_comparison.csv` - Comparison of all tested models

### After `python src/predict.py`:

**Console Output:**
```
✓ Model loaded: models/best_model.pkl
✓ Label encoder loaded: models/label_encoder.pkl
✓ Loaded 8 test samples from data/test.csv
✓ Test data preprocessing completed

GENERATING PREDICTIONS
✓ Predictions generated: 8 samples
  Predicted classes: ['ingredient_only', 'ingredient_with_qty', 
                      'instruction_like', 'non_food']

CLASSIFICATION METRICS
Accuracy: 0.7500
Macro F1 Score: 0.7500
Weighted F1 Score: 0.7500

Per-class metrics:
  ingredient_only      - Prec: 1.000, Rec: 0.500, F1: 0.667, Support: 2
  ingredient_with_qty  - Prec: 1.000, Rec: 1.000, F1: 1.000, Support: 2
  instruction_like     - Prec: 0.500, Rec: 1.000, F1: 0.667, Support: 2
  non_food            - Prec: 1.000, Rec: 0.500, F1: 0.667, Support: 2
```

**Generated Files:**
- `outputs/predictions.csv` - Test predictions with format:
  ```
  text,pred
  Garlic,ingredient_only
  Sugar 20 g,ingredient_with_qty
  ...
  ```
- `outputs/test_metrics.json` - Test set metrics in JSON format

### After `pytest tests/ -v`:

**Console Output:**
```
tests/test_preprocessing.py::TestTextCleaning::test_clean_text_lowercase PASSED
tests/test_preprocessing.py::TestTextCleaning::test_clean_text_punctuation_removal PASSED
...
tests/test_predict.py::TestMetricsComputation::test_f1_score_multiclass PASSED

======================= 60 passed in 2.34s =======================
```

---

## Customization & Configuration

### Preprocessing Configuration (src/train.py)

```python
PYCARET_CONFIG = {
    'normalize': True,              # Apply normalization
    'remove_stopwords': True,       # Remove English stopwords
    'remove_outliers': True,        # Remove extreme samples
    'outliers_threshold': 0.05,     # Threshold for outlier detection
    'n_jobs': -1,                   # -1 for all CPU cores
    'seed': 42,                     # Random seed for reproducibility
    'verbose': False,               # Show detailed training output
}
```

### Adding New Verbs for Instruction Detection

Edit `TextPreprocessor.INSTRUCTION_VERBS` in `src/preprocessing.py`:

```python
INSTRUCTION_VERBS = {
    'chop', 'dice', 'slice', ...,
    'your_new_verb'  # Add here
}
```

### Changing Cross-Validation Folds

In `src/train.py`, modify the `fold` parameter in `exp.setup()`:

```python
exp.setup(
    ...,
    fold=5,  # Change from 10 to 5
    ...
)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | Run `pip install -r requirements.txt` again |
| NLTK resource errors | Run `python -c "import nltk; nltk.download('punkt', 'stopwords')"` |
| Model not found | Ensure you've run `python src/train.py` first |
| Memory errors | Reduce `n_jobs` from -1 to 1 in PYCARET_CONFIG |
| PyCaret GPU errors | Set `gpu_param=False` in PyCaret setup |
| Prediction shape mismatch | Ensure preprocessing is consistent between training and inference |

---

## Performance Expectations

With 12 training samples and 8 test samples:

**Training Time:** 10-30 seconds
- PyCaret testing ~10-15 algorithms in 10-fold CV
- Total: ~100-150 model trainings

**Prediction Time:** <1 second
- Preprocessing + inference on 8 samples

**Memory Usage:** 100-500 MB
- Depends on chosen algorithm (Random Forest uses more than Logistic Regression)

**Model Size:** 5-50 MB (pickled)
- Ensemble models (RF, GB) larger than linear models

---

## Next Steps for Production

1. **Collect More Data:** Aim for 100+ samples per class
2. **Monitor Predictions:** Track confidence scores and errors
3. **A/B Testing:** Compare with alternative models
4. **Retraining Pipeline:** Automate monthly/weekly model updates
5. **API Deployment:** Wrap `predict.py` with FastAPI/Flask
6. **Model Monitoring:** Track drift in prediction distribution
7. **Explainability:** Add feature importance visualization
