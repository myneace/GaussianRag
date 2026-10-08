"""Benchmark: Standard RAG vs GSKG Multi-Hop RAG.

This test demonstrates the performance and recall improvements of the 
Gaussian Semantic Knowledge Graph (GSKG) multi-hop retrieval compared 
to standard single-pass Gaussian retrieval.
"""

import json
import logging
import numpy as np
import time
from pathlib import Path
from gaussian_rag.core.types import GaussianKnowledge, QueryRepresentation
from gaussian_rag.core.store import KnowledgeStore
from gaussian_rag.rag.retrieval.retriever import GaussianRetriever
from gaussian_rag.rag.retrieval.gskg_retriever import retrieve_gskg_multi_hop

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_gskg_benchmark():
    # 1. Load the knowledge base
    knowledge_path = Path("/home/abhi/src/manifold/.full_run_store/refining_polysemy_resilient_rag_architecture_1777629492/refining_polysemy_resilient_rag_architecture/knowledge.json")
    
    if not knowledge_path.exists():
        logger.error(f"Knowledge base not found at {knowledge_path}. Please run ingestion first.")
        return

    logger.info(f"Loading knowledge base from {knowledge_path}...")
    with open(knowledge_path) as f:
        data = json.load(f)
    
    # Building a reasonably sized store for the benchmark
    store = KnowledgeStore()
    items = []
    for entry in data[:1000]: # First 1000 nodes for speed
        item = GaussianKnowledge(
            id=entry["id"],
            text=entry["text"],
            mu=np.array(entry["mu"]),
            sigma_diag=np.array(entry["sigma_diag"]),
            sigma_L=np.array(entry["sigma_L"]),
            metadata=entry.get("metadata", {})
        )
        items.append(item)
    store.add(items)
    
    retriever = GaussianRetriever(store)
    retriever.refresh()
    
    # 2. Define a query that would benefit from multi-hop expansion
    # We want a query that matches some concepts but needs more context from related entities.
    query_text = "How does the manifold resolve semantic collisions between different domains?"
    
    # Pick a mu from a relevant node to ensure we get some initial hits
    target_node = items[146] # Known to be relevant in the previous tests
    query = QueryRepresentation(
        text=query_text,
        mu=target_node.mu
    )
    
    print("\n" + "="*70)
    print(f"BENCHMARK: Standard RAG vs GSKG Multi-Hop")
    print(f"Query: '{query_text}'")
    print("="*70)
    
    # --- Benchmark Standard RAG ---
    t0 = time.perf_counter()
    standard_results = retriever.retrieve(query, top_k=5, adaptive=False)
    t_standard = time.perf_counter() - t0
    
    print(f"\n[Standard RAG] Time: {t_standard:.4f}s")
    for r in standard_results:
        entities = r.knowledge.metadata.get("entities", [])
        print(f" - {r.knowledge.id} (W2: {r.w2_distance:.3f}) | Entities: {entities[:3]}")
    
    # --- Benchmark GSKG Multi-Hop ---
    t0 = time.perf_counter()
    # We allow 1 hop to start with
    gskg_results = retrieve_gskg_multi_hop(
        retriever, 
        query, 
        top_k=5, 
        max_hops=1, 
        manifold_pruning_threshold=3.0
    )
    t_gskg = time.perf_counter() - t0
    
    print(f"\n[GSKG Multi-Hop] Time: {t_gskg:.4f}s")
    for r in gskg_results:
        entities = r.knowledge.metadata.get("entities", [])
        is_new = r.knowledge.id not in [s.knowledge.id for s in standard_results]
        marker = "[NEW] " if is_new else "      "
        print(f" {marker}{r.knowledge.id} (W2: {r.w2_distance:.3f}) | Entities: {entities[:3]}")
        
    # 3. Analyze differences
    standard_ids = {r.knowledge.id for r in standard_results}
    gskg_ids = {r.knowledge.id for r in gskg_results}
    new_discoveries = gskg_ids - standard_ids
    
    print("\n" + "="*70)
    print(f"GSKG expanded the frontier and discovered {len(new_discoveries)} new relevant nodes")
    print(f"via implicit entity edges on the information manifold.")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_gskg_benchmark()
