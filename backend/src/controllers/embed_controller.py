from fastapi import UploadFile
import os
from src.utility.logger import get_logger
from src.utility.data_loader import process_and_generate_embeddings
from src.utility.vector_database import initialize_database, insert_product

logger = get_logger(__name__)

async def save_temp_file(file: UploadFile) -> str:
    """
    Save uploaded file temporarily.
    
    Args:
        file: Uploaded file object
        
    Returns:
        Path to the temporary file
    """
    contents = await file.read()
    temp_file = "temp/temp_products.csv"
    with open(temp_file, "wb") as f:
        f.write(contents)
    return temp_file

def process_and_insert_products(file_path: str) -> dict:
    """
    Process product data and insert into database.
    
    Args:
        file_path: Path to the product CSV file
        
    Returns:
        Dictionary containing processing results
    """
    # Process the data and generate embeddings
    logger.info(f"Processing uploaded CSV file: {file_path}")
    data = process_and_generate_embeddings(file_path)
    
    # Initialize Qdrant database
    initialize_database()
    
    # Insert products into Qdrant
    success_count = 0
    
    for index, row in data.iterrows():
        try:
            insert_product(
                product_id=int(index),  # Convert to integer
                description=row["merged_text"],
                embedding=row["embedding"]
            )
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to insert product {index}: {e}")
            
    return {
        "total_products": len(data),
        "successful_inserts": success_count
    }

def cleanup_temp_file(file_path: str):
    """
    Clean up temporary file after processing.
    
    Args:
        file_path: Path to the temporary file
    """
    try:
        os.remove(file_path)
    except Exception as e:
        logger.warning(f"Failed to remove temporary file: {e}")