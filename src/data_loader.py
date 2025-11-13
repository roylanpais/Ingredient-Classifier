import pandas as pd
from pathlib import Path
import logging
from src import config

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_train_data(path: Path = config.TRAIN_FILE) -> pd.DataFrame:
    """
    Loads the training data from the specified CSV file.

    Args:
        path: Path to the train.csv file.

    Returns:
        A pandas DataFrame with the training data.
    """
    try:
        df = pd.read_csv(path)
        log.info(f"Training data loaded successfully from {path}")
        
        # Basic validation
        if config.TEXT_COLUMN not in df.columns or config.TARGET_COLUMN not in df.columns:
            raise ValueError(f"Training data must contain '{config.TEXT_COLUMN}' and '{config.TARGET_COLUMN}' columns.")
        
        # Drop rows with missing text
        df = df.dropna(subset=[config.TEXT_COLUMN, config.TARGET_COLUMN])
        return df
        
    except FileNotFoundError:
        log.error(f"Error: Training file not found at {path}")
        raise
    except Exception as e:
        log.error(f"Error loading training data: {e}")
        raise

def load_test_data(path: Path = config.TEST_FILE) -> pd.DataFrame:
    """
    Loads the test data from the specified CSV file.

    Args:
        path: Path to the test.csv file.

    Returns:
        A pandas DataFrame with the test data.
    """
    try:
        df = pd.read_csv(path)
        log.info(f"Test data loaded successfully from {path}")
        
        # Basic validation
        if config.TEXT_COLUMN not in df.columns:
            raise ValueError(f"Test data must contain a '{config.TEXT_COLUMN}' column.")
        
        # Fill NA in text column with empty string to avoid prediction errors
        df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].fillna("")
        
        return df
        
    except FileNotFoundError:
        log.error(f"Error: Test file not found at {path}")
        raise
    except Exception as e:
        log.error(f"Error loading test data: {e}")
        raise