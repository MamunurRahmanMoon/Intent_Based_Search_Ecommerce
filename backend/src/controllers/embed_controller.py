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
    temp_file = "temp_products.csv"
    with open(temp_file, "wb") as f:
        f.write(contents)
    return temp_file

def process_and_insert_products(data) -> dict:
    """
    Process product data and insert into database.

    Args:
        data: Either a file path (str) or a pandas DataFrame containing product data

    Returns:
        Dictionary containing processing results
    """
    import pandas as pd
    from src.utility.embedding_model import EmbeddingModel

    # If data is a file path, load it
    if isinstance(data, str):
        logger.info(f"Processing uploaded CSV file: {data}")
        data = pd.read_csv(data)
    else:
        logger.info("Processing DataFrame")

    # Initialize Qdrant database
    initialize_database()
    model = EmbeddingModel()

    # Track already inserted pair_ids to avoid duplicates in this import session
    inserted_pair_ids = set()

    # Process each offer in each pair
    success_count = 0
    total_products = 0
    imported_products = 0

    logger.info(f"Starting to process {len(data)} rows")

    for idx, row in data.iterrows():
        if imported_products >= 10:
            break

        # Extract both left and right fields for payload
        fields = {}
        for side in ["_left", "_right"]:
            fields[f"title{side}"] = str(row.get(f"title{side}", ""))
            fields[f"description{side}"] = str(row.get(f"description{side}", ""))
            fields[f"brand{side}"] = str(row.get(f"brand{side}", ""))
            fields[f"category{side}"] = str(row.get(f"category{side}", ""))

        for offer_prefix in ["_left", "_right"]:
            if imported_products >= 10:
                break
            id_col = f"id{offer_prefix}"
            if id_col not in row or pd.isna(row[id_col]):
                logger.debug(f"Skipping row {idx} with prefix {offer_prefix}: No ID")
                continue

            product_id = int(row[id_col])
            title = fields[f"title{offer_prefix}"]
            description = fields[f"description{offer_prefix}"]
            brand = fields[f"brand{offer_prefix}"]
            category = fields[f"category{offer_prefix}"]

            logger.debug(f"Processing product {product_id} with title: {title}")

            merged_text = f"{title} {description} {brand} {category}"
            embedding = model.get_embedding(merged_text)

            # Payload includes both left and right data
            payload = {
                "pair_id": row.get("pair_id", ""),
                "title_left": fields["title_left"],
                "title_right": fields["title_right"],
                "description_left": fields["description_left"],
                "description_right": fields["description_right"],
                "brand_left": fields["brand_left"],
                "brand_right": fields["brand_right"],
                "category_left": fields["category_left"],
                "category_right": fields["category_right"]
            }

            pair_id = payload["pair_id"]
            if pair_id in inserted_pair_ids:
                logger.debug(f"Skipping duplicate pair_id {pair_id}")
                continue

            try:
                insert_product(
                    product_id=product_id,
                    description=merged_text,
                    embedding=embedding,
                    payload=payload
                )
                inserted_pair_ids.add(pair_id)
                success_count += 1
                imported_products += 1
                logger.debug(f"Successfully inserted product {product_id}")
            except Exception as e:
                logger.error(f"Failed to insert product {product_id}: {e}")
            total_products += 1

    return {
        "total_products": total_products,
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