from fastapi import APIRouter
from src.logger import get_logger

router = APIRouter()

# Initialize logger
logger = get_logger(__name__)

@router.get("/")
def root():
    return {
        "app_name": os.getenv("APP_NAME"),
        "message": "Welcome to the Intent-Based Search API"
    }

@router.get("/health")
def health_check():
    return {"status": "ok"}