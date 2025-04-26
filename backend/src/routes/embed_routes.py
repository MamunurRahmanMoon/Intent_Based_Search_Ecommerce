from fastapi import APIRouter, UploadFile, File
from src.controllers.embed_controller import save_temp_file, process_and_insert_products, cleanup_temp_file
from src.utility.logger import get_logger
from datasets import load_dataset
import tempfile
import os

logger = get_logger(__name__)

router = APIRouter(prefix="/embed", tags=["Embed"])


@router.post("")
async def embed_to_vector():
    """
    Embed the product data from a CSV file.
    """
    try:
        dataset = load_dataset("wdc/products-2017", "cameras_small")
        df = dataset["test"].to_pandas()

        # Save DataFrame to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as tmp:
            df.to_csv(tmp.name, index=False)
            temp_file_path = tmp.name

        # Process and insert products
        results = process_and_insert_products(df)

        # Optionally, clean up the temp file
        os.remove(temp_file_path)

        return {
            "status": "success",
            "message": f"Successfully processed and inserted {results['successful_inserts']} products",
            "total_products": results['total_products'],
            "successful_inserts": results['successful_inserts']
        }
    except Exception as e:
        logger.error(f"Error during embedding: {e}")
        return {
            "status": "error",
            "message": str(e)
        }