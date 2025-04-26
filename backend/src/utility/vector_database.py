# src/utility/vector_database.py
import os
from typing import List
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.models import PointStruct
from dotenv import load_dotenv
from src.utility.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# Initialize Qdrant client
client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    check_compatibility=False,  # Disable version check to avoid warnings
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
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        logger.info(f"Collection '{collection_name}' created in Qdrant.")
        logger.error(f"Error while checking or creating collection: {e}")


def insert_product(product_id: int, description: str, embedding: np.ndarray, payload: dict):
    """Insert a new product into the Qdrant collection, checking for duplicates"""
    collection_name = os.getenv("QDRANT_COLLECTION", "ecommerce")
    try:
        # Check for similar products (cosine similarity threshold of 0.8)
        # search_results = search_similar_products(embedding, top_k=1)
        # if search_results and search_results[0]["score"] > 0.8:
        #     existing_product = search_results[0]
        #     logger.info(f"Skipping duplicate product. Similar product found: {existing_product}")
        #     return False  # Return False to indicate duplicate

        # client.upsert(
        #     collection_name=collection_name,
        #     points=[
        #         {
        #             "id": product_id,  # Ensure this is an integer
        #             "vector": embedding.tolist(),
        #             "payload": {"description": description},
        #         }
        #     ],
        # )
        # payload = {"description": description}
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=product_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )
        logger.info(f"Successfully inserted product {product_id}")
        return True  # Return True to indicate successful insertion
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
                "payload": result.payload,
                "score": result.score,
            }
            for result in results
        ]
    except Exception as e:
        logger.error(f"Error during search in collection '{collection_name}': {e}")
        return []
