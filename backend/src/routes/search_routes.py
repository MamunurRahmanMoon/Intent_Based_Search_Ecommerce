from fastapi import APIRouter, UploadFile, File
from src.controllers.embed_controller import save_temp_file, process_and_insert_products, cleanup_temp_file
from src.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/embed", tags=["Embed"])

@router.post("")  # Changed from get to post since we're uploading a file
async def train(file: UploadFile = File(...)):
    """
    Train the model with new product data from a CSV file.
    
    Args:
        file: CSV file containing product data with columns: title, description
        
    Returns:
        Training status and summary
    """
    try:
        # Save file temporarily
        temp_file = await save_temp_file(file)
        
        # Process and insert products
        results = process_and_insert_products(temp_file)
        
        # Clean up temporary file
        cleanup_temp_file(temp_file)
        
        return {
            "status": "success",
            "message": f"Successfully processed and inserted {results['successful_inserts']} products",
            "total_products": results['total_products'],
            "successful_inserts": results['successful_inserts']
        }
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return {
            "status": "error",
            "message": str(e)
        }