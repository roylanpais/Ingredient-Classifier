# **Technical Decisions**

This document outlines the key technical decisions made during the development of the ingredient-classifier pipeline.

### **1\. Core Library: PyCaret**

* **Requirement**: "low-code library for testing multiple ML models."  
* **Decision**: PyCaret was chosen as the core library.  
* **Justification**: PyCaret's compare\_models function is the most direct and robust solution for this requirement. It provides a single command to train, cross-validate, and rank multiple models on a given dataset. This avoids writing extensive boilerplate code for each model.

### **2\. Module: pycaret.classification vs. pycaret.nlp**

* **Decision**: Use the pycaret.classification module, not pycaret.nlp.  
* **Justification**: The task is a supervised **classification** task, not an unsupervised NLP task.  
  * The pycaret.nlp module is designed for unsupervised learning (e.g., Topic Modeling, Word Embeddings).  
  * The pycaret.classification module fully supports text-based classification by specifying text columns in the text\_features parameter of the setup() function. It automatically handles NLP preprocessing (e.g., TF-IDF vectorization, feature engineering) as part of the classification pipeline.

### **3\. Metrics Calculation**

* **Requirement**: "add classification metrics calculation (include Macro F1) on the test set."  
* **Issue**: The provided test.csv file does not contain ground-truth labels, making it impossible to calculate metrics on that specific file.  
* **Decision**: Metrics are calculated on the 30% hold-out set that PyCaret automatically splits from the train.csv data.  
* **Justification**: This is standard machine learning practice. The hold-out set (or validation set) serves as an unseen proxy for the test set to evaluate the model's generalization performance.  
  * The compare\_models function is configured to sort='F1', which for multiclass problems defaults to Macro F1.  
  * The full metrics dataframe (including Macro F1) from this hold-out set is captured using pull() and saved to outputs/metrics.json.

### **4\. Model Selection**

* **Decision**: The pipeline compares a curated list of models: \['lr', 'nb', 'ridge', 'svm', 'lightgbm'\].  
* **Justification**: These models are fast to train and traditionally perform very well on text classification (TF-IDF) tasks.  
  * lr (Logistic Regression) and nb (Naive Bayes) are strong, classic baselines.  
  * ridge (Ridge Classifier) and svm (Linear SVM) are also powerful and efficient for high-dimensional sparse data like text features.  
  * lightgbm (LightGBM) is included as a high-performance tree-based model.  
    The model with the highest Macro F1 score on the hold-out set is automatically selected, finalized (trained on 100% of train.csv), and saved.

### **5\. Testing Strategy**

* **test\_pipeline.py (Integration Test)**: This test ensures the end-to-end pipeline works. It creates a temporary project structure, runs the train\_model and generate\_predictions functions, and asserts that all expected files (.pkl model, .json metrics, .csv predictions) are created in the correct format.  
* **test\_features.py (Unit Test)**: This test focuses on model robustness. It loads the model trained by the test fixture and feeds it edge cases (e.g., empty strings, numbers-only, punctuation, unseen examples). The test asserts that the model does not crash and returns a valid label from the configured list of classes. This verifies graceful handling of unexpected input.