import pytest
import pandas as pd
from pathlib import Path
from src import config

@pytest.fixture(scope="module")
def setup_test_environment(tmp_path_factory):
    """
    Pytest fixture to create a temporary project structure for testing.
    
    - Creates temp dirs for data, models, outputs.
    - Creates mock train.csv and test.csv files.
    - Monkeypatches the src.config module to use these temp paths.
    - Yields the base temporary directory.
    """
    
    # 1. Create temporary directories
    base_dir = tmp_path_factory.mktemp("project")
    data_dir = base_dir / "data"
    model_dir = base_dir / "models"
    output_dir = base_dir / "outputs"
    data_dir.mkdir()
    model_dir.mkdir()
    output_dir.mkdir()

    # 2. Create mock train.csv (based on user-provided snippet)
    train_data = {
        "text": [
            "Tomato", "Onion", "Butter 50 g", "Milk 200 ml", 
            "Chop the onions", "Simmer for 10 minutes", "Plastic wrap", 
            "Baking paper", "Eggs 2", "Olive oil", "Slice the bread", "Paper towel",
            # Add more data for robustness
            "Salt", "Pepper 5g", "Stir gently", "Spatula"
        ],
        "label": [
            "ingredient_only", "ingredient_only", "ingredient_with_qty", "ingredient_with_qty",
            "instruction_like", "instruction_like", "non_food",
            "non_food", "ingredient_with_qty", "ingredient_only", "instruction_like", "non_food",
            "ingredient_only", "ingredient_with_qty", "instruction_like", "non_food"
        ]
    }
    df_train = pd.DataFrame(train_data)
    mock_train_path = data_dir / "train.csv"
    df_train.to_csv(mock_train_path, index=False)

    # 3. Create mock test.csv (based on user-provided snippet)
    test_data = {
        "text": [
            "Garlic", "Sugar 20 g", "Warm in a pan", "Aluminum foil",
            "Rice 150 g", "Cumin", "Stir for 2 minutes", "Salt"
        ],
        "label": ["", "", "", "", "", "", "", ""] # Empty as in original
    }
    df_test = pd.DataFrame(test_data)
    mock_test_path = data_dir / "test.csv"
    df_test.to_csv(mock_test_path, index=False)

    # 4. Monkeypatch config module to use temp paths
    #    We save original paths to restore them later, though for 'module' scope
    #    it's less critical, it's good practice.
    original_paths = {
        "TRAIN_FILE": config.TRAIN_FILE,
        "TEST_FILE": config.TEST_FILE,
        "MODEL_DIR": config.MODEL_DIR,
        "MODEL_PATH": config.MODEL_PATH,
        "OUTPUT_DIR": config.OUTPUT_DIR,
        "PREDICTION_FILE": config.PREDICTION_FILE,
        "METRICS_FILE": config.METRICS_FILE,
    }

    config.TRAIN_FILE = mock_train_path
    config.TEST_FILE = mock_test_path
    config.MODEL_DIR = model_dir
    config.MODEL_PATH = model_dir / config.MODEL_NAME
    config.OUTPUT_DIR = output_dir
    config.PREDICTION_FILE = output_dir / "predictions.csv"
    config.METRICS_FILE = output_dir / "metrics.json"

    yield base_dir  # This is where the tests will run

    # 5. Teardown: Restore original config paths
    for key, value in original_paths.items():
        setattr(config, key, value)