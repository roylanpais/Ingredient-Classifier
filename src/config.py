from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).parent.parent.resolve()

# Data directories
DATA_DIR = BASE_DIR / "data"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"

# Output directories
OUTPUT_DIR = BASE_DIR / "outputs"
PREDICTION_FILE = OUTPUT_DIR / "predictions.csv"
METRICS_FILE = OUTPUT_DIR / "metrics.json"

# Model directories
MODEL_DIR = BASE_DIR / "models"
MODEL_NAME = "ingredient_classifier"
MODEL_PATH = MODEL_DIR / MODEL_NAME

# Classification details
TARGET_COLUMN = "label"
TEXT_COLUMN = "text"

# List of valid classes for validation
CLASSES = [
    "ingredient_only",
    "ingredient_with_qty",
    "instruction_like",
    "non_food"
]

# Models to compare.
# We choose fast, strong text classifiers.
MODELS_TO_COMPARE = ['lr', 'nb', 'ridge', 'svm', 'lightgbm']