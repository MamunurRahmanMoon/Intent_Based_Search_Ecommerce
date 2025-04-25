# src/data_loader.py
import pandas as pd
import os
from src.embedding_model import EmbeddingModel
from src.logger import get_logger

logger = get_logger(__name__)


def load_csv(file_path):
    """Load product data from a CSV file."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        logger.info(f"Loading CSV file from {file_path}.")
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error while loading CSV file: {e}")
        raise


def process_and_generate_embeddings(file_path):
    """Load CSV, merge title and description, generate embeddings, and return processed data."""
    try:
        data = load_csv(file_path)
        model = EmbeddingModel()
        # Merge title and description columns
        data["merged_text"] = data["title"] + " " + data["description"]
        logger.info("Merged 'title' and 'description' columns.")
        # Generate embeddings for the merged text
        data["embedding"] = data["merged_text"].apply(model.get_embedding)
        logger.info("Generated embeddings for the merged text.")
        return data
    except Exception as e:
        logger.error(f"Error during data processing or embedding generation: {e}")
        raise
