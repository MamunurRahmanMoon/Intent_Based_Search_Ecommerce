import os
from typing import List
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from dotenv import load_dotenv
from src.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# Initialize Qdrant client
client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY"),
)


def initialize_database():
    """Initialize the Qdrant collection"""
    collection_name = os.getenv("QDRANT_COLLECTION", "ecommerce")
    try:
        # Check if the collection exists
        client.get_collection(collection_name)
        logger.info(f"Collection '{collection_name}' already exists in Qdrant.")
    except Exception as e:
        # Create the collection if it does not exist
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        logger.info(f"Collection '{collection_name}' created in Qdrant.")
        logger.error(f"Error while checking or creating collection: {e}")


def insert_product(product_id: int, description: str, embedding: np.ndarray):
    """Insert a new product into the Qdrant collection"""
    collection_name = os.getenv("QDRANT_COLLECTION", "ecommerce")
    try:
        client.upsert(
            collection_name=collection_name,
            points=[
                {
                    "id": product_id,  # Ensure this is an integer
                    "vector": embedding.tolist(),
                    "payload": {"description": description},
                }
            ],
        )
        logger.info(f"Successfully inserted product {product_id}")
    except Exception as e:
        logger.error(f"Error while inserting product {product_id}: {e}")
        raise


def search_similar_products(query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
    """Search for similar products using Qdrant"""
    collection_name = os.getenv("QDRANT_COLLECTION", "ecommerce")
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
        )
        logger.info(
            f"Search completed in collection '{collection_name}' for top {top_k} results."
        )
        return [
            {
                "product_id": result.id,
                "description": result.payload["description"],
                "score": result.score,
            }
            for result in results
        ]
    except Exception as e:
        logger.error(f"Error during search in collection '{collection_name}': {e}")
        return []
