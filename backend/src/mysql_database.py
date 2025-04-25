import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
from src.logger import get_logger

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
