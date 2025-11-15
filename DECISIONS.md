# Architecture Decisions & Trade-offs

## Overview
This document outlines key architectural and methodological decisions made during development, their rationale, trade-offs, and alternatives considered.

---

## 1. Low-Code Approach with PyCaret

### Decision
Use **PyCaret** for automated model selection and hyperparameter tuning.

### Rationale
- **Efficiency**: Tests multiple algorithms (Logistic Regression, Random Forest, Gradient Boosting, SVM, Naive Bayes, KNN, etc.) in minimal code
- **Reduced complexity**: Handles preprocessing, feature scaling, and cross-validation automatically
- **Production-ready**: Built-in utilities for model saving, prediction, and metrics
- **Quick iteration**: Ideal for small datasets (12 training samples) where manual model tuning would be premature

### Trade-offs
| Trade-off | Impact |
|-----------|--------|
| Less control over preprocessing | Mitigated by custom preprocessing module that can be extended |
| Binary reproducibility issues with some ensemble methods | Addressed by setting `seed` parameter |
| Potential overfitting on small datasets | Controlled via cross-validation (PyCaret default: 10-fold) |

### Alternatives Considered
1. **Pure scikit-learn**: Offers more control but requires manual model testing and tuning (verbose for small datasets)
2. **AutoML tools (H2O, TPOT)**: More expensive computationally; unnecessary for this problem size

### Recommendation for Scaling
- If dataset grows >100K samples, transition to custom scikit-learn pipeline with explicit hyperparameter optimization (GridSearchCV/RandomizedSearchCV)

---

## 2. Feature Engineering Strategy

### Decision
Implement **lightweight rule-based + statistical features**:
- Text length metrics (character count, word count)
- Digit presence (indicator of quantities)
- Verb detection (heuristic for instructions)
- TF-IDF vectorization (standard text representation)

### Rationale
- **Interpretability**: Features align with class semantics:
  - `ingredient_with_qty`: High digit presence
  - `instruction_like`: High verb count
  - Text length helps distinguish classes
- **Computational efficiency**: No deep learning needed for small dataset
- **Robustness**: Explicit features prevent model from learning spurious patterns

### Trade-offs
| Feature Type | Pros | Cons |
|-------------|------|------|
| Rule-based (verbs, digits) | Interpretable, fast | May miss edge cases, language-dependent |
| TF-IDF | Captures semantic patterns | Generic; doesn't exploit domain knowledge |
| Deep embeddings (BERT) | State-of-the-art | Overkill for 12 samples; prone to overfitting |

### Examples
```
"Tomato" → features: [len=6, words=1, has_digits=0, has_verbs=0, tfidf_vector]
"Milk 200 ml" → features: [len=11, words=3, has_digits=1, has_verbs=0, tfidf_vector]
"Chop the onions" → features: [len=15, words=3, has_digits=0, has_verbs=1, tfidf_vector]
```

### Alternative Considered
- **Embeddings (Word2Vec, FastText)**: Would require external pre-trained models or large corpus; not suitable for this dataset size

---

## 3. Data Preprocessing Pipeline

### Decision
Implement **modular preprocessing** with these steps:
1. Lowercasing
2. Punctuation removal
3. Optional stop word removal (configurable)
4. Tokenization (NLTK-based)
5. Lemmatization option

### Rationale
- **Modularity**: Each step can be toggled via configuration
- **Normalization**: Reduces feature space noise (e.g., "Chop" vs "chop")
- **Reproducibility**: Fixed preprocessing order ensures consistent results

### Trade-offs
| Decision | Trade-off |
|----------|-----------|
| Remove stopwords | Removes noise (better for small datasets) vs. loses contextual info (rare for ingredient domain) |
| Lemmatization | Reduces feature dimensionality vs. loses word form information |

### Configuration in `train.py`
```python
PYCARET_CONFIG = {
    'remove_stopwords': True,      # Toggle stop word removal
    'normalize': True,              # Lowercasing + punctuation
}
```

### Alternative Considered
- **No preprocessing**: Would introduce noise and sparse features; not suitable

---

## 4. Train/Test Split & Validation Strategy

### Decision
**No explicit train/test split on training data**; use PyCaret's **stratified K-Fold cross-validation (10 folds)**.

### Rationale
- **Dataset size**: Only 12 training samples; cannot afford to hold out 20-30% for validation
- **Stratification**: Ensures each fold has balanced class distribution
- **Small dataset handling**: K-fold provides reliable performance estimates
- **Test set**: External test set (8 samples) used for final evaluation

### Validation Pipeline
```
Train (12 samples)
    ↓
PyCaret 10-Fold CV
    ↓
Best model selected
    ↓
Test set evaluation (8 samples)
```

### Trade-offs
| Approach | Pros | Cons |
|----------|------|------|
| Hold-out split | Simple, faster | Loses 2-3 samples; unreliable with n=12 |
| K-Fold CV | Robust estimates | Computationally expensive (10 models trained) |
| Leave-One-Out CV | Maximum data usage | Expensive with non-trivial datasets |

---

## 5. Evaluation Metrics

### Decision
Report **Macro F1 Score** as primary metric, plus:
- Per-class Precision & Recall
- Weighted F1 Score
- Confusion Matrix

### Rationale
- **Macro F1**: Fair for imbalanced classes (e.g., if one class has fewer samples)
- **Per-class metrics**: Identifies which classes are misclassified
- **Confusion matrix**: Reveals systematic errors (e.g., instruction_like vs. ingredient_with_qty)

### Calculations
```
Macro F1 = (F1_ingredient_only + F1_ingredient_with_qty + F1_instruction_like + F1_non_food) / 4

Where: F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### Trade-offs
| Metric | Use Case | Trade-off |
|--------|----------|-----------|
| Macro F1 | Imbalanced data | Ignores class weights |
| Weighted F1 | Realistic performance | Less interpretable for small classes |
| Accuracy | Simplicity | Misleading with imbalanced data |

---

## 6. Model Serialization & Persistence

### Decision
Use **pickle** for model serialization with version tracking.

### Rationale
- **Built-in support**: PyCaret models and scikit-learn pipelines are pickle-compatible
- **Fast loading**: No serialization overhead
- **Label encoding preservation**: Custom label encoder pickled alongside model

### Limitations & Mitigations
| Limitation | Mitigation |
|-----------|-----------|
| Python version dependency | Document Python version (3.8+) in requirements |
| Not interoperable with other languages | Use ONNX export if cross-language support needed |
| Large model files | Acceptable for this project size |

### File Structure
```
models/
├── best_model.pkl           # PyCaret best model
└── label_encoder.pkl        # LabelEncoder for class names
```

### Alternative Considered
- **ONNX format**: Language-agnostic; overkill for internal use
- **JSON + weights**: Manual serialization; error-prone

---

## 7. Testing Strategy

### Decision
Implement **unit tests** with pytest covering:
- Preprocessing edge cases (empty strings, special characters, extremes)
- Text feature extraction correctness
- Model training pipeline
- Inference on unseen data

### Test Categories
| Test Type | Purpose | Examples |
|-----------|---------|----------|
| Preprocessing | Validate data cleaning | Empty string, special chars, unicode |
| Feature extraction | Verify feature computation | Digit detection, verb count, length |
| Pipeline | End-to-end training flow | Training completes, model saves |
| Inference | Prediction correctness | Batch prediction, single sample |

### Trade-offs
| Decision | Trade-off |
|----------|-----------|
| Comprehensive unit tests | Requires maintenance effort vs. ensures reliability |
| Mock external dependencies | Faster tests vs. less realistic coverage |
| Use fixtures for data | Cleaner code vs. test data management |

---

## 8. Project Structure & Modularity

### Decision
Organize code into **three modules**:
- `preprocessing.py` – Text cleaning and feature engineering
- `train.py` – Model selection and training (PyCaret-based)
- `predict.py` – Inference and prediction export

### Rationale
- **Separation of concerns**: Each module has single responsibility
- **Reusability**: Preprocessing can be used independently
- **Testability**: Each module can be unit tested in isolation
- **Maintainability**: Clear dependencies; easy to debug

### Module Dependencies
```
predict.py
    ↓ imports
train.py, preprocessing.py
    ↓ imports
pandas, sklearn, pycaret, nltk
```

---

## 9. Reproducibility & Documentation

### Rationale
- **Reproducibility**: Any user can rebuild the environment identically
- **Transparency**: Stakeholders understand design choices
- **Maintenance**: Future updates easier with clear documentation

### Versioning Strategy
- **Pinned versions**: Ensures identical behavior across runs and environments
- **NLTK resources**: Explicitly downloaded in setup.sh
- **Seed setting**: Fixed random seed in PyCaret for deterministic model selection

---

## 10. Error Handling & Logging

### Decision
Implement **defensive programming**:
- Type hints and input validation
- Try-catch blocks for external operations (file I/O, model loading)
- Informative error messages

### Examples
```python
def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV with validation."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError("Dataset is empty")
    return df
```

### Trade-offs
| Approach | Pros | Cons |
|----------|------|------|
| Strict validation | Prevents silent failures | Verbose code |
| Loose validation | Simpler code | Harder to debug |

---

## Known Limitations & Future Work

### Current Limitations
1. **Small dataset (12 samples)**: High variance in CV metrics; model may overfit
2. **Language-specific**: Verb lists hardcoded for English
3. **No class weights**: Assumes balanced data (actual: 3 ingredient_only, 3 ingredient_with_qty, 3 instruction_like, 3 non_food)
4. **No hyperparameter tuning**: Uses PyCaret defaults

### Recommended Improvements
1. **Collect more data**: Aim for 100+ samples per class for robust evaluation
2. **Domain-specific embeddings**: Train Word2Vec on recipe corpus
3. **Multilingual support**: Extend preprocessing for other languages
4. **Model monitoring**: Track prediction confidence and drift in production
5. **A/B testing**: Compare against baseline/alternative models in production

---

## Summary Table

| Component | Decision | Rationale |
|-----------|----------|-----------|
| Model Selection | PyCaret | Automated testing of multiple algorithms |
| Features | Rule-based + TF-IDF | Domain-aware + statistical representation |
| Preprocessing | Modular pipeline | Interpretable, configurable |
| Validation | Stratified K-Fold CV | Robust with small datasets |
| Metrics | Macro F1 + per-class | Fair evaluation for imbalanced classes |
| Serialization | Pickle | Compatible with PyCaret & scikit-learn |
| Testing | Unit tests (pytest) | Ensures reliability & maintainability |
| Reproducibility | setup.sh + requirements.txt | Identical environment across machines |

