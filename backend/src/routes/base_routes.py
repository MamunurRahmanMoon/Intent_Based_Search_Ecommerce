from fastapi import APIRouter
from src.utility.logger import get_logger
import os

# Initialize logger
logger = get_logger(__name__)

router = APIRouter(prefix="", tags=["Base"])

@router.get("/")
def root():
    return {
        "app_name": os.getenv("APP_NAME"),
        "message": "Welcome to the Intent-Based Search API"
    }

@router.get("/health")
def health_check():
    return {"status": "ok"}