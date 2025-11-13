import typer
import logging
from src import pipeline
from src import config

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = typer.Typer(
    help="CLI for the Ingredient Classification pipeline."
)

@app.command()
def train():
    """
    Trains the classifier on data/train.csv.
    Saves the model to models/ and metrics to outputs/
    """
    log.info("Starting 'train' command...")
    # Ensure directories exist
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    pipeline.train_model()

@app.command()
def predict():
    """
    Uses the trained model to generate predictions on data/test.csv.
    Saves results to outputs/predictions.csv.
    """
    log.info("Starting 'predict' command...")
    pipeline.generate_predictions()

if __name__ == "__main__":
    app()