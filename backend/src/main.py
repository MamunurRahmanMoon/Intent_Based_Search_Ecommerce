# src/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from src.utility.vector_database import search_similar_products, initialize_database, insert_product
from src.utility.embedding_model import EmbeddingModel
from src.utility.logger import get_logger
from src.utility.data_loader import process_and_generate_embeddings
from src.routes import embed_routes, base_router
import os
import pandas as pd

app = FastAPI()

# Initialize the embedding model
model = EmbeddingModel()

# Initialize logger
logger = get_logger(__name__)

app.include_router(embed_routes.router)
app.include_router(base_router)

# class SearchRequest(BaseModel):
#     query: str
#     top_k: int = 5

# @app.post("/search")
# def search(request: SearchRequest):
#     """Perform semantic search on the product database."""
#     try:
#         logger.info(
#             f"Received search request with query: {request.query} and top_k: {request.top_k}"
#         )
#         # Generate embedding for the query
#         query_embedding = model.get_embedding(request.query)
#         logger.info("Generated embedding for the query.")
#         # Perform search in Qdrant
#         results = search_similar_products(query_embedding, top_k=request.top_k)
#         logger.info("Search completed successfully.")
#         return {"results": results}
#     except Exception as e:
#         logger.error(f"Error during search: {e}")
#         raise HTTPException(
#             status_code=500, detail="An error occurred during the search."
#         )
