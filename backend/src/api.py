# src/search_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.vector_database import search_similar_products
from .embedding_model import EmbeddingModel
from src.logger import get_logger

import os

app = FastAPI()

# Initialize the embedding model
model = EmbeddingModel()

# Initialize logger
logger = get_logger(__name__)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def root():
    return {
        "app_name": os.getenv("APP_NAME"),
        "message": "Welcome to the Intent-Based Search API"
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/search")
def search(request: SearchRequest):
    """Perform semantic search on the product database."""
    try:
        logger.info(
            f"Received search request with query: {request.query} and top_k: {request.top_k}"
        )
        # Generate embedding for the query
        query_embedding = model.get_embedding(request.query)
        logger.info("Generated embedding for the query.")
        # Perform search in Qdrant
        results = search_similar_products(query_embedding, top_k=request.top_k)
        logger.info("Search completed successfully.")
        return {"results": results}
    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred during the search."
        )
