from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
from gaussian_rag.core.types import GaussianKnowledge, QueryRepresentation, RetrievedChunk
from gaussian_rag.core.information_geometry import wasserstein2, fisher_rao
from gaussian_rag.evaluation.stratifier import ambiguity_bucket
from gaussian_rag.evaluation.metrics import ndcg_at_k, recall_at_k

@dataclass
class ComparisonResult:
    method: str
    bucket: str
    ndcg: float
    recall: float

class PointRetriever:
    """Baseline: Standard Point RAG using Dot Product (Cosine Similarity)."""
    def __init__(self, knowledge: List[GaussianKnowledge]):
        self.knowledge = knowledge
        self.mu_matrix = np.vstack([k.mu for k in knowledge])
        self.ids = [k.id for k in knowledge]

    def retrieve(self, query_mu: np.ndarray, top_k: int = 3) -> List[str]:
        # Mu is already normalized in GaussianEmbedder
        scores = np.dot(self.mu_matrix, query_mu)
        indices = np.argsort(scores)[::-1][:top_k]
        return [self.ids[i] for i in indices]

class ManifoldRetriever:
    """Gaussian RAG using Wasserstein-2 distance."""
    def __init__(self, knowledge: List[GaussianKnowledge]):
        self.knowledge = knowledge

    def retrieve(self, query: QueryRepresentation, top_k: int = 3) -> List[str]:
        q_gaussian = query.as_gaussian()
        distances = []
        for k in self.knowledge:
            distances.append((k.id, wasserstein2(q_gaussian, k)))
        distances.sort(key=lambda x: x[1])
        return [id for id, _ in distances[:top_k]]

class FisherRaoRetriever:
    """Gaussian RAG using Fisher-Rao metric."""
    def __init__(self, knowledge: List[GaussianKnowledge]):
        self.knowledge = knowledge

    def retrieve(self, query: QueryRepresentation, top_k: int = 3) -> List[str]:
        q_gaussian = query.as_gaussian()
        distances = []
        for k in self.knowledge:
            distances.append((k.id, fisher_rao(q_gaussian, k)))
        distances.sort(key=lambda x: x[1])
        return [id for id, _ in distances[:top_k]]

def run_experiment():
    embedder = GaussianEmbedder(sigma_mode="heuristic")
    
    # Define a corpus that has distinct "certain" docs and "ambiguous" docs
    # Doc 1: Highly specific medical fact
    # Doc 2: Broad, ambiguous statement about law
    # Doc 3: Highly specific legal procedure
    # Doc 4: Ambiguous statement about science uncertainty
    corpus_texts = [
        ("m1", "Mitral valve regurgitation describes backward blood flow from the left ventricle to the left atrium."),
        ("l1", "Rule 11 sanctions may be imposed if a pleading is submitted for an improper purpose like harassment."),
        ("a1", "Uncertainty in data interpretation often arises from systemic bias and low sample sizes in clinical trials."),
        ("a2", "Ambiguous legal precedents lead to conflicting judicial outcomes in different jurisdictions."),
    ]
    
    # Scenario 2: Semantic Collision (Same Mean, Different Variance)
    # Doc S1: "treatment of X" (Certain)
    # Doc S2: "treatment of X is unknown and varies wildly" (Ambiguous)
    # Query: "treatment for X is uncertain" 
    # Point RAG might rank S1 higher if it has more matches.
    # Manifold RAG should favor S2 if the query variance matches the doc variance.
    
    collision_corpus = [
        ("s1", "The treatment for condition X is clearly defined as drug A."),
        ("s2", "The treatment for condition X is currently uncertain and can vary significantly based on patient data."),
    ]
    
    collision_kb = [embedder.encode(text, item_id=id) for id, text in collision_corpus]
    collision_point = PointRetriever(collision_kb)
    collision_manifold = ManifoldRetriever(collision_kb)
    
    collision_bench = [
        ("uncertain treatment for condition X with significant variation", ["s2"]),
    ]
    
    print("\nSemantic Collision Scenario:")
    print(f"{'Method':<15} | {'Query':<30} | {'Top Result':<10} | {'Correct?'}")
    print("-" * 70)
    for q_text, rel in collision_bench:
        # Point
        p_res = collision_point.retrieve(embedder._encode_vector(q_text), top_k=1)
        print(f"{'point':<15} | {q_text[:30]:<30} | {p_res[0]:<10} | {p_res[0] in rel}")
        # Manifold
        q_rep = embedder.encode_query(q_text, with_covariance=True)
        m_res = collision_manifold.retrieve(q_rep, top_k=1)
        print(f"{'manifold_w2':<15} | {q_text[:30]:<30} | {m_res[0]:<10} | {m_res[0] in rel}")
    
    # Scenario 3: Theoretical Manifold Advantage (Synthetic Vectors)
    dim = 2
    q_mu = np.array([1.0, 0.0])
    q_sigma_diag = np.array([0.5, 0.5]) # Ambiguous query
    
    # doc_wrong: point-wise perfect match, but very "certain" (narrow)
    w_mu = np.array([1.0, 0.0])
    w_sigma_diag = np.array([0.001, 0.001])
    
    # doc_right: point-wise further, but "ambiguous" (matches query spread)
    r_mu = np.array([0.80, 0.0])
    r_sigma_diag = np.array([0.5, 0.5])
    
    def _to_kb(id, mu, sd):
        return GaussianKnowledge(id=id, text="synthetic", mu=mu, sigma_diag=sd, sigma_L=np.zeros((dim, 0)))

    syn_kb = [
        _to_kb("wrong", w_mu, w_sigma_diag),
        _to_kb("right", r_mu, r_sigma_diag),
    ]
    
    # For this synthetic test, we use a custom Manifold Ranker that doesn't divide by confidence
    # to show the pure geometric distance advantage.
    def pure_w2_ranker(q_rep, kb):
        q_g = q_rep.as_gaussian()
        dists = [(k.id, wasserstein2(q_g, k)) for k in kb]
        return sorted(dists, key=lambda x: x[1])

    syn_point = PointRetriever(syn_kb)
    
    print("\nTheoretical Manifold Advantage (Synthetic Vectors):")
    print(f"{'Method':<15} | {'Top Result':<10} | {'Distance/Score'}")
    print("-" * 60)
    # Point
    p_top = syn_point.retrieve(q_mu, top_k=1)[0]
    p_score = np.dot(w_mu, q_mu) if p_top == "wrong" else np.dot(r_mu, q_mu)
    print(f"{'point':<15} | {p_top:<10} | {p_score:.4f} (higher is better)")
    
    # Manifold
    q_rep = QueryRepresentation(text="syn_q", mu=q_mu, sigma_diag=q_sigma_diag, sigma_L=np.zeros((dim, 0)))
    m_ranked = pure_w2_ranker(q_rep, syn_kb)
    m_top = m_ranked[0][0]
    print(f"{'manifold_w2':<15} | {m_top:<10} | {m_ranked[0][1]:.4f} (lower is better)")
    
    knowledge_base = [embedder.encode(text, item_id=id) for id, text in corpus_texts]
    
    point_retriever = PointRetriever(knowledge_base)
    manifold_retriever = ManifoldRetriever(knowledge_base)
    fr_retriever = FisherRaoRetriever(knowledge_base)
    
    # Benchmarks: (query, relevant_ids)
    # Queries designed to vary in lexical entropy (ambiguity)
    benchmarks = [
        # Low Ambiguity (Specific)
        ("mitral valve regurgitation ventricle blood flow atrium", ["m1"]),
        ("Rule 11 sanctions legal harassment purpose improper", ["l1"]),
        
        # Medium Ambiguity
        ("legal precedent interpret context venue jurisdiction outcome", ["a2", "l1"]),
        ("scientific study uncertainty bias data clinical trials sample size", ["a1"]),
        
        # High Ambiguity (Broad/Confusing/Longer)
        ("The ambiguity of legal precedents leads to conflicting outcomes across many jurisdictions and interpretations of the law.", ["a2"]),
        ("High uncertainty in scientific data interpretation often arises from systemic bias and extremely low sample sizes in clinical trials with conflicting studies.", ["a1"]),
        ("What are the outcomes when data interpretation in clinical trials is ambiguous and leads to jurisdictional conflicts in legal settings?", ["a1", "a2"]),
    ]
    
    results: List[ComparisonResult] = []
    
    for query_text, relevant_ids in benchmarks:
        bucket = ambiguity_bucket(query_text)
        
        # 1. Point RAG
        q_mu = embedder._encode_vector(query_text)
        point_ids = point_retriever.retrieve(q_mu, top_k=2)
        results.append(ComparisonResult("point", bucket, ndcg_at_k(point_ids, relevant_ids, 2), recall_at_k(point_ids, relevant_ids, 2)))
        
        # 2. Manifold RAG (W2)
        q_rep = embedder.encode_query(query_text, with_covariance=True)
        manifold_ids = manifold_retriever.retrieve(q_rep, top_k=2)
        results.append(ComparisonResult("manifold_w2", bucket, ndcg_at_k(manifold_ids, relevant_ids, 2), recall_at_k(manifold_ids, relevant_ids, 2)))
        
        # 3. Fisher-Rao
        fr_ids = fr_retriever.retrieve(q_rep, top_k=2)
        results.append(ComparisonResult("manifold_fr", bucket, ndcg_at_k(fr_ids, relevant_ids, 2), recall_at_k(fr_ids, relevant_ids, 2)))

    # Aggregating results by method and bucket
    summary: Dict[str, Dict[str, List[float]]] = {}
    for res in results:
        key = f"{res.method}|{res.bucket}"
        if key not in summary:
            summary[key] = {"ndcg": [], "recall": []}
        summary[key]["ndcg"].append(res.ndcg)
        summary[key]["recall"].append(res.recall)
        
    print(f"{'Method':<15} | {'Bucket':<10} | {'nDCG@2':<8} | {'Recall@2':<8}")
    print("-" * 50)
    for key, vals in sorted(summary.items()):
        method, bucket = key.split("|")
        m_ndcg = np.mean(vals["ndcg"])
        m_recall = np.mean(vals["recall"])
        print(f"{method:<15} | {bucket:<10} | {m_ndcg:<8.3f} | {m_recall:<8.3f}")

if __name__ == "__main__":
    run_experiment()
