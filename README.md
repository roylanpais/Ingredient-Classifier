# **Ingredient Classifier**

This project contains a production-ready machine learning pipeline to classify short ingredient text lines into one of four categories:

* ingredient\_only  
* ingredient\_with\_qty  
* instruction\_like  
* non\_food

The pipeline is built using PyCaret to leverage its low-code, high-performance environment for rapidly training, comparing, and deploying a classification model.

## **Features**

* **Low-Code Model Comparison**: Uses PyCaret to automatically preprocess text data and compare multiple ML models (Logistic Regression, Naive Bayes, LightGBM, etc.) to find the best one.  
* **Reproducible Pipeline**: Full setup script (run.sh) to create a virtual environment, install dependencies, and run the entire pipeline.  
* **Production-Ready Structure**: Code is separated into modules for configuration, data loading, and pipeline logic.  
* **CLI Interface**: Uses typer for clean train and predict commands.  
* **Testing**: Includes unit tests with pytest to verify pipeline integrity and robustness against edge cases.  
* **Detailed Outputs**: Generates predictions.csv for the test set and metrics.json with performance (including Macro F1) from the hold-out validation set.

## **Project Structure**

ingredient-classifier/  
|-- data/  
|   |-- train.csv         (Input training data)  
|   |-- test.csv          (Input test data for prediction)  
|-- models/  
|   |-- .gitkeep          (Stores the serialized model)  
|-- outputs/  
|   |-- .gitkeep          (Stores predictions and metrics)  
|-- src/  
|   |-- \_\_init\_\_.py  
|   |-- config.py         (All file paths and constants)  
|   |-- data\_loader.py    (Functions to load CSV data)  
|   |-- pipeline.py       (Core logic for training and prediction)  
|   |-- main.py           (CLI entry point using Typer)  
|-- tests/  
|   |-- \_\_init\_\_.py  
|   |-- conftest.py       (Pytest fixture for setting up test environment)  
|   |-- test\_pipeline.py  (Integration tests for train/predict)  
|   |-- test\_features.py  (Unit tests for edge cases)  
|-- .gitignore  
|-- DECISIONS.md          (Explanation of technical choices)  
|-- requirements.txt      (Python dependencies)  
|-- run.sh                (Main execution script)

## **Setup & Running**

**Note:** This guide assumes your uploaded train.csv and test.csv files are located in the directory *above* the ingredient-classifier/ project folder.

1. **Clone the project** (or in this case, create the directory and files as provided).  
2. **Make the run script executable:**  
   chmod \+x run.sh

3. Run the script:  
   This script will do everything:  
   * Create a Python virtual environment (venv/).  
   * Activate it.  
   * Install all required dependencies from requirements.txt.  
   * Create the data/, models/, and outputs/ directories.  
   * Copy your train.csv and test.csv into the data/ directory.  
   * Run the pytest suite.  
   * Run the train command (training and saving the model).  
   * Run the predict command (generating predictions.csv).

./run.sh

## **Pipeline Output**

After running, you will find two new files in the outputs/ directory:

1. **metrics.json**: A JSON file containing the full performance metrics (Accuracy, F1, Kappa, etc.) of the best-performing model on the internal 30% hold-out set from train.csv.  
2. **predictions.csv**: A CSV file with the original text from test.csv and the model's pred (predicted label).