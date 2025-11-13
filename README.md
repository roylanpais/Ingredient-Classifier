# Ingredient Line Classifier
This project uses pycaret to train a multi-class text classifier to categorize ingredient lines.
The goal is to classify short text snippets into one of four categories:
ingredient_only — e.g., "Tomato"
ingredient_with_qty — e.g., "Milk 200 ml"
instruction_like — e.g., "Chop the onions"
non_food — e.g., "Plastic wrap"
Project Structure
ingredient_classifier/
|
├── data/
│   ├── train.csv         # Training data with labels
│   └── test.csv          # Test data without labels
|
├── .gitignore
├── DECISIONS.md        # Documentation of design choices
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── run_project.sh      # End-to-end script for setup, train, predict, test
├── train_predict.py    # Main Python script for training and prediction
|
└── tests/
    └── test_classifier.py # Integration tests for model behavior


Setup
Create and activate a Python virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install the required dependencies:
pip install -r requirements.txt


Running the Project
You can run the entire workflow (install, train, predict, test) using the shell script.
bash run_project.sh


Manual Steps
If you prefer to run the steps manually:
Train the Model:
This will load data/train.csv, find the best model, train it on the full dataset, and save it as ingredient_model.pkl.
python train_predict.py --mode train


Generate Predictions:
This will load the saved ingredient_model.pkl, predict on data/test.csv, and save the results to predictions.csv.
python train_predict.py --mode predict


Run Tests:
This will load the saved model and run integration tests against specific edge cases.
pytest


Output
The script run_project.sh or python train_predict.py --mode predict will generate:
ingredient_model.pkl: The saved, trained PyCaret model pipeline.
predictions.csv: A CSV file with columns text and pred containing the model's predictions for data/test.csv.
