# src/search_api.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from src.vector_database import search_similar_products, initialize_database, insert_product
from .embedding_model import EmbeddingModel
from src.logger import get_logger
from src.data_loader import process_and_generate_embeddings
import os
import pandas as pd

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

@app.post("/embed")
async def train(file: UploadFile = File(...)):
    # print(file)
    # return {
    #     "status": "success",
    #     "message": "Coming Soon",
    #     "total_products": 0,
    #     "successful_inserts": []
    # }

    """Train the model with new product data from a CSV file.
    
    Args:
        file: CSV file containing product data with columns: title, description
    
    Returns:
        Training status and summary
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        # Save temporarily
        temp_file = "temp_products.csv"
        with open(temp_file, "wb") as f:
            f.write(contents)
            
        # Process the data and generate embeddings
        logger.info(f"Processing uploaded CSV file: {file.filename}")
        data = process_and_generate_embeddings(temp_file)
        
        # Initialize Qdrant database
        initialize_database()
        
        # Insert products into Qdrant
        # collection_name = os.getenv("QDRANT_COLLECTION", "ecommerce")
        success_count = 0
        
        for index, row in data.iterrows():
            try:
                insert_product(
                    product_id=str(index),
                    description=row["merged_text"],
                    embedding=row["embedding"]
                )
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to insert product {index}: {e}")
                
        # Clean up temporary file
        os.remove(temp_file)
        
        return {
            "status": "success",
            "message": f"Successfully processed and inserted {success_count} products",
            "total_products": len(data),
            "successful_inserts": success_count
        }
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

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
