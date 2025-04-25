import logging
import os

# Create a logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),  # Log to a file
        logging.StreamHandler(),  # Log to the console
    ],
)


# Get a logger instance
def get_logger(name):
    return logging.getLogger(name)
