from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from src.utility.logger import get_logger
from src.controllers.search_controller import hybrid_search, initialize_search

logger = get_logger(__name__)

# Create a FastAPI router for search endpoints
router = APIRouter(prefix="/search", tags=["Search"])

# Initialize search components when the app starts
@router.on_event("startup")
def initialize_search_on_startup():
    """Initialize search components when the application starts."""
    initialize_search()

# Request model for search
class SearchRequest(BaseModel):
    query: str  # Search query
    top_k: int = 5  # Number of results to return
    semantic_weight: Optional[float] = 0.7  # Weight for semantic score in hybrid search

# Response model for search results
class SearchResult(BaseModel):
    id: int
    score: float
    payload: dict
    source: Optional[str] = None

# Hybrid search endpoint (only one endpoint for simplicity)
@router.post("", response_model=List[SearchResult])
async def search_products(request: SearchRequest):
    """
    Perform a hybrid search using both vector (Qdrant) and BM25 (text) search.
    """
    try:
        logger.info(
            f"Received hybrid search request with query: {request.query}, top_k: {request.top_k}"
        )
        # Call the hybrid search function
        results = hybrid_search(request.query, request.top_k, request.semantic_weight)
        logger.info(f"Hybrid search completed successfully. Found {len(results)} results")
        return results
    except HTTPException as e:
        logger.error(f"HTTP error during search: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))