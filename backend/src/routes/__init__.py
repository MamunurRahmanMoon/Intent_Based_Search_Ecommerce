from .embed_routes import router as embed_router
from .base_routes import router as base_router
from .search_routes import router as search_router

__all__ = ['embed_router', 'base_router', 'search_router']
