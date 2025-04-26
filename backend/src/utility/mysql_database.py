# src/utility/mysql_database.py
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
from src.utility.logger import get_logger
from typing import List, Dict, Any

# Load environment variables from .env
load_dotenv()

logger = get_logger(__name__)


def initialize_mysql_database():
    """Initialize the MySQL database and table."""
    connection = None  # Initialize connection to avoid UnboundLocalError
    try:
        # Connect to MySQL server
        connection = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
        )

        if connection.is_connected():
            cursor = connection.cursor()

            # Create database if it doesn't exist
            cursor.execute("CREATE DATABASE IF NOT EXISTS ecommerce;")
            logger.info("Database 'ecommerce' created or already exists.")

            # Use the database
            cursor.execute("USE ecommerce;")

            # Create table if it doesn't exist
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS products (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    merged_text TEXT NOT NULL
                );
                """
            )
            logger.info("Table 'products' created or already exists.")

    except Error as e:
        logger.error(f"Error while connecting to MySQL: {e}")

    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            logger.info("MySQL connection closed.")


def get_db_connection():
    """Get a MySQL database connection."""
    try:
        return mysql.connector.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "ecommerce")
        )
    except Exception as e:
        logger.error(f"Error connecting to MySQL: {e}")
        raise


def search_products_bm25(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform BM25 search on the products table.
    
    Args:
        query: Search query
        top_k: Number of results to return
        
    Returns:
        List of search results with BM25 scores
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Split query into terms
        terms = query.split()
        
        # Build BM25 query
        query_parts = []
        for term in terms:
            query_parts.append(f"MATCH(title, description) AGAINST(%s IN BOOLEAN MODE)")
        
        query_str = " AND ".join(query_parts)
        
        # Execute query
        cursor.execute(f"""
            SELECT 
                id,
                title,
                description,
                (SUM({query_str})) as score
            FROM products
            WHERE {query_str}
            GROUP BY id
            ORDER BY score DESC
            LIMIT %s
        """, terms + [top_k])
        
        results = cursor.fetchall()
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result["id"],
                "score": float(result["score"]),
                "payload": {
                    "title": result["title"],
                    "description": result["description"]
                }
            })
        
        cursor.close()
        conn.close()
        
        logger.info(f"BM25 search completed. Found {len(formatted_results)} results")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error during BM25 search: {e}")
        raise
