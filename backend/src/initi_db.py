import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mysql_database import initialize_mysql_database

# Initialize the MySQL database and table
initialize_mysql_database()

from src.vector_database import initialize_database

# Initialize the Qdrant collection
initialize_database()

from src.vector_database import insert_product
from src.data_loader import process_and_generate_embeddings

file_path = os.path.join("data", "raw", "product.csv")
data = process_and_generate_embeddings(file_path)

for _, row in data.iterrows():
    product_id = row.name  # Use the row index as the product ID
    description = row["description"]
    embedding = row["embedding"]
    insert_product(product_id, description, embedding)
