# src/embedding_model.py
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """A class to generate embeddings using a pre-trained model."""

    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for the given text."""
        return np.array(self.model.encode(text))
