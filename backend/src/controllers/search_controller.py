from typing import Optional, List, Dict, Any
from src.utility.logger import get_logger
from src.utility.embedding_model import EmbeddingModel
from src.utility.vector_database import search_similar_products, client
from src.utility.bm25_search import initialize_bm25, search_products_bm25
import numpy as np
import os

logger = get_logger(__name__)
model = EmbeddingModel()

def initialize_search():
    """
    Initialize search components.
    """
    try:
        # Load product data from Qdrant
        collection_name = os.getenv("QDRANT_COLLECTION", "ecommerce")
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=1000  # Adjust limit based on your dataset size
        )
        
        # Extract text data for BM25
        corpus = []
        for point in points:
            text = point.payload.get("description", "")
            if text:
                corpus.append(text)
        
        # Initialize BM25 with the corpus
        initialize_bm25(corpus)
        logger.info("BM25 initialized successfully with %d documents", len(corpus))
        
    except Exception as e:
        logger.error(f"Error initializing search: {e}")
        raise

def semantic_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform semantic search using embeddings.
    """
    try:
        logger.info(f"Performing semantic search for query: {query}")
        query_embedding = model.get_embedding(query)
        logger.info("Generated embedding for the query.")
        
        results = search_similar_products(query_embedding, top_k=top_k)
        logger.info(f"Semantic search completed successfully. Found {len(results)} results")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during semantic search: {e}")
        raise

def bm25_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform BM25 search.
    """
    try:
        logger.info(f"Performing BM25 search for query: {query}")
        results = search_products_bm25(query, top_k=top_k)
        logger.info(f"BM25 search completed successfully. Found {len(results)} results")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during BM25 search: {e}")
        raise

def hybrid_search(query: str, top_k: int = 5, semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining semantic (vector) and BM25 (text) results.
    If BM25 finds no products, return only semantic results.
    """
    try:
        logger.info(f"Performing hybrid search for query: {query}")
        semantic_results = semantic_search(query, top_k=top_k)
        bm25_results = bm25_search(query, top_k=top_k)

        # Build dictionaries for fast lookup
        bm25_dict = {r.get("id"): r for r in bm25_results if r.get("id") is not None}
        semantic_dict = {r.get("id"): r for r in semantic_results if r.get("id") is not None}

        # --- Hybrid (BM25 + Semantic) search following Qdrant/BM25 demo logic ---
        # 1. Build BM25 score map for all valid doc ids
        bm25_score_map = {}
        for r in bm25_results:
            pid = r.get("id")
            if pid is not None:
                bm25_score_map[pid] = float(r.get("score", 0))

        # 2. Build semantic (vector) score map for all valid doc ids
        sem_score_map = {}
        for r in semantic_results:
            pid = r.get("id")
            if pid is not None:
                sem_score_map[pid] = float(r.get("score", 0))

        # 3. Normalize scores (min-max normalization for fair hybridization)
        def min_max_norm(scores):
            if not scores:
                return {}
            values = list(scores.values())
            min_v, max_v = min(values), max(values)
            if max_v == min_v:
                return {k: 0.0 for k in scores}  # avoid div by zero
            return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}

        bm25_norm = min_max_norm(bm25_score_map)
        sem_norm = min_max_norm(sem_score_map)

        # 4. Union all doc ids from both sources, skipping None
        all_ids = set(bm25_score_map.keys()).union(set(sem_score_map.keys()))
        alpha = semantic_weight  # user can tune this
        combined_results = []
        for pid in all_ids:
            if pid is None:
                continue
            hybrid_score = bm25_norm.get(pid, 0.0) * (1 - alpha) + sem_norm.get(pid, 0.0) * alpha
            # Prefer semantic payload if available, else bm25
            payload = None
            if pid in sem_score_map:
                payload = semantic_dict.get(pid, {}).get("payload", {})
            elif pid in bm25_score_map:
                payload = bm25_dict.get(pid, {}).get("payload", {})
            combined_results.append({
                "id": int(pid),
                "score": hybrid_score,
                "payload": payload,
                "source": "hybrid"
            })
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"Hybrid search completed successfully. Found {len(combined_results[:top_k])} results")
        return combined_results[:top_k]

    except Exception as e:
        logger.error(f"Error during hybrid search: {e}")
        raise