from typing import Optional, List, Dict, Any
from src.utility.logger import get_logger
from src.utility.embedding_model import EmbeddingModel
from src.utility.vector_database import search_similar_products, client
from src.utility.bm25_search import BM25_INSTANCE, search_products_bm25, initialize_bm25
from src.utility.intent_extractor import IntentExtractor
import numpy as np
import os

logger = get_logger(__name__)
model = EmbeddingModel()
intent_extractor = IntentExtractor()

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
        payloads = []
        for point in points:
            fields = []
            for key in [
                "title_left", "title_right",
                "description_left", "description_right",
                "brand_left", "brand_right",
                "category_left", "category_right"
            ]:
                value = point.payload.get(key, "")
                if value and value != "None":
                    fields.append(str(value))
            text = " ".join(fields).strip()
            if text:
                corpus.append(text)
                payloads.append(point.payload)
        
        if not corpus:
            logger.warning("No documents found to initialize BM25. Skipping BM25 initialization.")
            return
        
        # Initialize BM25 with the corpus and payloads
        initialize_bm25(corpus, payloads)
        logger.info("BM25 initialized successfully with %d documents", len(corpus))
        
    except Exception as e:
        logger.error(f"Error initializing search: {e}")
        raise

def bm25_search_with_lazy_init(query: str, top_k: int = 5):
    if BM25_INSTANCE is None:
        initialize_search()
    return search_products_bm25(query, top_k=top_k)

def semantic_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform semantic search using embeddings, analyzing intent first.
    """
    try:
        # 1. Extract intent
        intent = intent_extractor.extract_intent_components(query)
        logger.info(f"Intent extracted: {intent}")

        # 2. Continue with embedding and search
        logger.info(f"Performing semantic search for query: {query}")
        query_embedding = model.get_embedding(query)
        logger.info("Generated embedding for the query.")
        
        results = search_similar_products(query_embedding, top_k=top_k)
        logger.info(f"Semantic search completed successfully. Found {len(results)} results")

        # 3. Intent-based filtering (demo: filter by constraints, e.g., price)
        filtered_results = results
        if intent.get("constraints"):
            constraints = intent["constraints"]
            filtered_results = []
            for result in results:
                payload = result.get("payload", {})
                # Example: filter for price constraints like 'under $500'
                for constraint in constraints:
                    if "under $" in constraint.lower():
                        try:
                            max_price = float(constraint.lower().split("under $")[-1].replace(",", "").strip())
                            price = float(payload.get("price", 0))
                            if price > 0 and price < max_price:
                                filtered_results.append(result)
                        except Exception:
                            continue
            # If no results matched constraints, fallback to original results
            if not filtered_results:
                filtered_results = results
        
        # Attach intent to response for transparency
        return {"intent": intent, "results": filtered_results}
        
    except Exception as e:
        logger.error(f"Error during semantic search: {e}")
        raise

def bm25_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform BM25 search.
    """
    try:
        logger.info(f"Performing BM25 search for query: {query}")
        results = bm25_search_with_lazy_init(query, top_k=top_k)
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
        bm25_results = bm25_search_with_lazy_init(query, top_k=top_k)

        # Build dictionaries for fast lookup
        bm25_dict = {r.get("id"): r for r in bm25_results if r.get("id") is not None}
        semantic_dict = {r.get("id"): r for r in semantic_results["results"] if r.get("id") is not None}

        # --- Hybrid (BM25 + Semantic) search following Qdrant/BM25 demo logic ---
        # 1. Build BM25 score map for all valid doc ids
        bm25_score_map = {}
        for r in bm25_results:
            pid = r.get("id")
            if pid is not None:
                bm25_score_map[pid] = float(r.get("score", 0))

        # 2. Build semantic (vector) score map for all valid doc ids
        sem_score_map = {}
        for r in semantic_results["results"]:
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

        # If BM25 has no data, return semantic results as hybrid
        if not bm25_score_map:
            logger.warning(f"semantic_results: {semantic_results}")
            combined_results = []
            for idx, result in enumerate(semantic_results["results"]):
                pid = result.get("id", idx)  # fallback to index if id is missing
                combined_results.append({
                    "id": int(pid) if pid is not None else idx,
                    "score": result.get("score", 0.0),
                    "payload": result.get("payload", {}),
                    "source": "hybrid"
                })
            combined_results.sort(key=lambda x: x["score"], reverse=True)
            logger.info(f"Hybrid search (semantic-only fallback). Found {len(combined_results[:top_k])} results")
            return combined_results[:top_k]

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