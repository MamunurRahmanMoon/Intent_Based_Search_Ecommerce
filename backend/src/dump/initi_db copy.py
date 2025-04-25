import sys
import os
import pandas as pd

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mysql_database import initialize_mysql_database
from src.vector_database import initialize_database, insert_product
from src.data_loader import process_and_generate_embeddings

def setup_database():
    """Initialize both MySQL and Qdrant databases."""
    # Initialize MySQL database
    initialize_mysql_database()
    
    # Initialize Qdrant collection
    initialize_database()

def load_product_data(file_path: str) -> pd.DataFrame:
    """
    Process and generate embeddings for product data.
    
    Args:
        file_path: Path to the product CSV file
        
    Returns:
        DataFrame containing processed product data with embeddings
    """
    return process_and_generate_embeddings(file_path)

def insert_products(data: pd.DataFrame):
    """
    Insert products into the database with their embeddings.
    
    Args:
        data: DataFrame containing product data with embeddings
    """
    for _, row in data.iterrows():
        product_id = row.name  # Use the row index as the product ID
        description = row["description"]
        embedding = row["embedding"]
        insert_product(product_id, description, embedding)

def main():
    """Main function to initialize database and load product data."""
    # Setup databases
    setup_database()
    
    # Define file path
    file_path = os.path.join("data", "raw", "product.csv")
    
    # Process and load data
    data = load_product_data(file_path)
    insert_products(data)

if __name__ == "__main__":
    main()
