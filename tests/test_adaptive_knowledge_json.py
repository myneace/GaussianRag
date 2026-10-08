import json
import pytest
import numpy as np
import logging
from pathlib import Path
from gaussian_rag.core.types import GaussianKnowledge, QueryRepresentation
from gaussian_rag.core.store import KnowledgeStore
from gaussian_rag.rag.retrieval.retriever import GaussianRetriever

# Setup detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_adaptive_retrieval_on_knowledge_base():
    logger.info("Starting Adaptive Retrieval Test")
    
    # 1. Load the knowledge base provided
    knowledge_path = Path("/home/abhi/src/manifold/.full_run_store/refining_polysemy_resilient_rag_architecture_1777629492/refining_polysemy_resilient_rag_architecture/knowledge.json")
    logger.info(f"Loading knowledge base from {knowledge_path}")
    
    with open(knowledge_path) as f:
        data = json.load(f)
        
    logger.info(f"Loaded {len(data)} total entries from JSON.")
    
    # Target to ensure it is in the subset
    target_id = "refining_polysemy_resilient_rag_architecture-chunk-146-aiprop-0"
    
    # Optimization: To prevent taking 10 minutes to calculate Wasserstein metrics across the entire 94MB dump,
    # we will load a subset of nodes, ensuring our target is included.
    logger.info("Building KnowledgeStore (parsing numpy arrays)...")
    store = KnowledgeStore()
    items = []
    
    for i, entry in enumerate(data):
        if i % 100 == 0:
            logger.info(f"Processed {i}/{len(data)} items...")
            
        item = GaussianKnowledge(
            id=entry["id"],
            text=entry["text"],
            mu=np.array(entry["mu"]),
            sigma_diag=np.array(entry["sigma_diag"]),
            sigma_L=np.array(entry["sigma_L"]),
            metadata={}
        )
        items.append(item)
    
    store.add(items)
    
    # 2. Setup the Adaptive Gaussian Retriever
    logger.info("Initializing GaussianRetriever and building MuAnnIndex...")
    retriever = GaussianRetriever(store)
    retriever.refresh()
    logger.info("Index build complete.")
    
    # 3. Simulate a query about the test being run
    target_item = next((item for item in items if item.id == target_id), None)
    assert target_item is not None, "Target item not found!"
    
    query = QueryRepresentation(
        text="Did the speaker run an initial test for the listener?",
        mu=target_item.mu
    )
    
    # 4. Perform an adaptive retrieval.
    logger.info("Executing adaptive retrieval. Requested top_k=1, adaptive=True, buffer_size=1")
    logger.info("The system should ignore top_k=1 and dynamically find the score gap...")
    results = retriever.retrieve(query, top_k=1, adaptive=True, buffer_size=1)
    
    # 5. Assertions and logging
    retrieved_ids = [r.knowledge.id for r in results]
    logger.info(f"Retrieval complete. Fetched {len(results)} chunks.")
    
    assert target_id in retrieved_ids, "Failed to retrieve the necessary target chunk."
    assert len(results) > 1, f"Expected adaptive retrieval to bypass top_k=1, but got {len(results)} results."
    
    print("\n" + "="*50)
    print(f"[Adaptive Retrieval Success] Question asked: '{query.text}'")
    print(f"Number of chunks adaptively retrieved: {len(results)} (bypassed top_k=1 limit)")
    for i, res in enumerate(results):
        print(f" - {i+1}. {res.knowledge.text[:60]}... (Dist: {res.w2_distance:.4f})")
    print("="*50 + "\n")


def test_mmasa_parallel_multi_sense():
    """Smoke test: MMASA parallel retriever produces the same fused result as the serial path."""
    import time

    knowledge_path = Path("/home/abhi/src/manifold/.full_run_store/refining_polysemy_resilient_rag_architecture_1777629492/refining_polysemy_resilient_rag_architecture/knowledge.json")
    logger.info("[MMASA Test] Loading knowledge subset (150 nodes)...")

    with open(knowledge_path) as f:
        data = json.load(f)

    # Small subset so the test finishes quickly
    items = [
        GaussianKnowledge(
            id=e["id"],
            text=e["text"],
            mu=np.array(e["mu"]),
            sigma_diag=np.array(e["sigma_diag"]),
            sigma_L=np.array(e["sigma_L"]),
            metadata={},
        )
        for e in data[:150]
    ]

    store = KnowledgeStore()
    store.add(items)

    retriever = GaussianRetriever(store)
    retriever.refresh()
    logger.info("[MMASA Test] Index ready. Building two senses...")

    # Two independent senses (simulating polysemy)
    sense_a = QueryRepresentation(text="How does retrieval work?", mu=items[0].mu)
    sense_b = QueryRepresentation(text="What is Wasserstein distance?", mu=items[10].mu)
    senses = [
        ("sense_a", 0.6, sense_a),
        ("sense_b", 0.4, sense_b),
    ]

    # --- Serial baseline ---
    logger.info("[MMASA Test] Running SERIAL multi-sense retrieval...")
    t0 = time.perf_counter()
    serial_provenance, serial_fused = retriever.retrieve_multi_sense(
        senses, top_k=3, adaptive=False
    )
    t_serial = time.perf_counter() - t0
    logger.info(f"[MMASA Test] Serial done in {t_serial:.3f}s  fused_nodes={len(serial_fused)}")

    # --- MMASA parallel ---
    logger.info("[MMASA Test] Running MMASA PARALLEL multi-sense retrieval...")
    t0 = time.perf_counter()
    parallel_provenance, parallel_fused = retriever.retrieve_multi_sense_parallel(
        senses, top_k=3, adaptive=False
    )
    t_parallel = time.perf_counter() - t0
    logger.info(f"[MMASA Test] Parallel done in {t_parallel:.3f}s  fused_nodes={len(parallel_fused)}")

    # Both paths must return the same fused node IDs
    serial_ids   = {c.knowledge.id for c in serial_fused}
    parallel_ids = {c.knowledge.id for c in parallel_fused}
    assert serial_ids == parallel_ids, (
        f"MMASA parallel result differs from serial!\n"
        f"  serial:   {serial_ids}\n"
        f"  parallel: {parallel_ids}"
    )

    # Both paths must have sense provenance keys
    assert set(parallel_provenance.keys()) == {"sense_a", "sense_b"}

    print("\n" + "="*60)
    print("[MMASA Smoke Test PASSED]")
    print(f"  Serial   wall-time : {t_serial:.3f}s")
    print(f"  Parallel wall-time : {t_parallel:.3f}s")
    speedup = t_serial / t_parallel if t_parallel > 0 else 1.0
    print(f"  Speedup            : {speedup:.2f}x")
    print(f"  Fused nodes        : {len(parallel_fused)}")
    print(f"  Sense A nodes      : {len(parallel_provenance['sense_a'])}")
    print(f"  Sense B nodes      : {len(parallel_provenance['sense_b'])}")
    print("="*60 + "\n")
