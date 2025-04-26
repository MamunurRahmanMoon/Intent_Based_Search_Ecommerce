from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src.utility.logger import get_logger
import numpy as np

logger = get_logger(__name__)

# Global variables to store corpus, payloads, and BM25 instance
CORPUS = []
BM25_PAYLOADS = []
BM25_INSTANCE = None

def initialize_bm25(corpus: List[str], payloads: List[dict]):
    """
    Initialize BM25 with the given corpus and store payloads for result lookup.
    """
    global CORPUS, BM25_INSTANCE, BM25_PAYLOADS
    CORPUS = corpus
    BM25_PAYLOADS = payloads
    # Tokenize the corpus for BM25
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    BM25_INSTANCE = BM25Okapi(tokenized_corpus)
    logger.info("BM25 initialized with corpus of size: %d", len(CORPUS))

def search_products_bm25(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform BM25 search on the products.
    """
    if BM25_INSTANCE is None:
        raise RuntimeError("BM25 not initialized. Please call initialize_bm25 first.")
    
    # Tokenize the query in the same way as the corpus
    tokenized_query = query.lower().split()
    scores = BM25_INSTANCE.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    # Format results
    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # Only include results with positive score
            results.append({
                "id": idx,
                "score": float(scores[idx]),
                "payload": BM25_PAYLOADS[idx]
            })
    
    logger.info("BM25 search completed. Found %d results", len(results))
    return results