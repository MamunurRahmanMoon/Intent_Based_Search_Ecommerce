import logging
import os
from datetime import datetime

# Create a logs directory if it doesn't exist
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Generate a log file name based on the current date and time
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),  # Log to a file with a dynamic name
        logging.StreamHandler(),  # Log to the console
    ],
)


# Get a logger instance
def get_logger(name):
    return logging.getLogger(name)
