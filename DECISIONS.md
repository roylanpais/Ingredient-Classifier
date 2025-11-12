Design & Architectural Decisions

This document outlines the key decisions made during the development of the ingredient line classifier.

1. Tooling: PyCaret

Decision: Use the pycaret[nlp] library as requested.

Rationale: PyCaret is a high-level, low-code library that automates many of the tedious ML steps. Its pycaret.nlp module is specifically designed for text classification tasks. It handles text preprocessing (tokenization, stop-word removal, stemming/lemmatization, TF-IDF vectorization) and model comparison in a single setup() and compare_models() workflow.

Trade-off: We are using classic ML models (e.g., Logistic Regression, Random Forest) on TF-IDF features. A more complex approach using fine-tuned Transformer models (like BERT) might yield higher accuracy but would be significantly more complex, slower, and resource-intensive. For short text lines, classic ML is a powerful and efficient baseline.

2. Model Selection & Metric

Decision: Use compare_models() to find the best-performing model, optimizing for Macro F1 Score.

Rationale:

compare_models() automatically trains and evaluates a wide range of classifiers, saving significant development time.

The target classes might be imbalanced (e.g., many ingredient_only lines, few non_food). Standard Accuracy can be misleading. Macro F1 calculates the F1 score for each class independently and then takes the unweighted average. This makes it a robust metric that ensures the model performs well on all classes, including the rare ones.

Implementation: We use add_metric('macro_f1', 'Macro F1', f1_score, average='macro') after setup() to add this metric to the scoring table, and then sort='Macro F1' in compare_models().

3. Validation Strategy

Decision: Use the 10-fold Cross-Validation (CV) results from compare_models() as the primary performance benchmark.

Rationale: The user asked for metrics on the "test set," but the provided test.csv has no labels. The standard, robust way to estimate a model's performance on unseen data is cross-validation.

Process:

The setup() function is called on the entire train.csv.

compare_models() runs a 10-fold CV for each algorithm. The printed table (which we pull() and display) shows the average Macro F1 (and other metrics) across these 10 folds. This is our trusted performance estimate.

finalize_model() is then called on the best model, which retrains it on the full train.csv dataset. This model, trained on 100% of the available labeled data, is what we save for production.

4. Testing Strategy

Decision: Implement integration tests (pytest) that load the saved production model and test its predictions on specific edge cases.

Rationale: Simple unit tests (e.g., "does this function return a string?") are less valuable than testing the actual behavior of the trained model. Our tests in tests/test_classifier.py provide confidence that the final artifact (ingredient_model.pkl) behaves as expected on tricky inputs.

Trade-off: This makes the tests dependent on the training step (python train_predict.py --mode train). This is a valid integration testing pattern and is automated in the run_project.sh script.

5. Scripting & Reproducibility

Decision: Separate training and prediction logic into one script (train_predict.py) controlled by an argparse flag (--mode).

Rationale: This follows a standard production pattern. You train a model artifact once, and then you predict with that artifact many times. This script cleanly separates these two distinct phases.

Automation: The run_project.sh script ensures full reproducibility. It creates a virtual environment, installs exact dependencies from requirements.txt, trains the model, runs predictions, and executes the tests, all in one command.
