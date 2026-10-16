"""test_48_laws_e2e.py -- End-to-end evaluation on The 48 Laws of Power corpus.

Pipeline stages:
1. Gaussian Knowledge Store load (cache-aware, skips re-embed if .48_laws_store/ exists)
2. Hierarchical Taxonomy build via TaxonomyRouter.build_from_store()
3. Clef Ambiguity Gate (noul question via request_decision)
4. Clef Taxonomy Branch Routing (choice question)
5. SQLite FTS5 BM25 Lexical Pre-filter
6. Clef 4-Question Passage Gate (concurrent batch)
7. Woodbury Low-Rank ELK Reranking
8. OpenAI-compatible LLM Final Answer Generation (query 1 only)
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gaussian_rag.config import load_project_config, load_dotenv_file, load_provider_env
from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
from gaussian_rag.core.lexical_store import extract_query_keywords
from gaussian_rag.core.store import KnowledgeStore
from gaussian_rag.core.types import GaussianKnowledge
from gaussian_rag.rag.disambiguation.clef_client import ClefClient, NoulQuestion
from gaussian_rag.rag.generation.generator import ProviderBackedGenerator
from gaussian_rag.rag.retrieval.clef_passage_gate import ClefPassageGate
from gaussian_rag.rag.retrieval.manifold_ranker import rerank_candidates
from gaussian_rag.rag.retrieval.retriever import GaussianRetriever
from gaussian_rag.rag.retrieval.taxonomy_router import TaxonomyRouter


def _ensure_env():
    env_file = _ROOT.parent / ".env"
    if env_file.exists():
        for k, v in load_dotenv_file(env_file).items():
            os.environ.setdefault(k, v)


def parse_48_laws(text_path: Path) -> list[tuple[int, str, str, str]]:
    """Return list of (law_num, title, heading, full_text) tuples."""
    raw = text_path.read_text(encoding="utf-8")
    sections = re.split(r"\n## Law (\d+): ([^\n]+)\n", raw)
    laws = []
    for i in range(1, len(sections), 3):
        if i + 2 < len(sections):
            num = int(sections[i])
            title = sections[i + 1].strip()
            body = sections[i + 2].strip()
            if "---" in body:
                body = body.split("---")[0].strip()
            heading = f"Law {num}: {title}"
            full_text = f"[{heading}] {body}"
            laws.append((num, title, heading, full_text))
    return laws


def assign_domain(num: int) -> str:
    if num in {1, 24, 34, 41, 46}:
        return "Courtiership and Superior Dynamics"
    elif num in {2, 8, 14, 15, 19, 22, 42, 44}:
        return "Adversaries Spies and Conflict Tactics"
    elif num in {3, 4, 5, 6, 12, 16, 21, 26, 30, 36, 38}:
        return "Deception Reputation and Strategic Secrecy"
    elif num in {7, 10, 11, 13, 20, 27, 31, 33, 40, 43}:
        return "Influence Coalitions and Human Leverage"
    else:
        return "Execution Timing and Adaptive Strategy"


async def main():
    _ensure_env()
    sep = "=" * 80
    print(sep)
    print("MANIFOLD GAUSSIAN-CLEF RAG: EVALUATION ON THE 48 LAWS OF POWER")
    print(sep)

    corpus_path = _ROOT / "data" / "48_laws_of_power.txt"
    if not corpus_path.exists():
        print(f"Error: {corpus_path} not found.")
        return

    laws = parse_48_laws(corpus_path)
    print(f"\n[1] Parsed {len(laws)} laws from corpus.")

    # Embedder always needed for query encoding
    embedder = GaussianEmbedder(
        dimension=384,
        sigma_mode="heuristic",
        rank=8,
        embedding_backend="local",
    )

    save_dir = _ROOT / ".48_laws_store"
    knowledge_cache = save_dir / "knowledge.json"

    # ---- Step 2: Load or Build KnowledgeStore (cache-aware) ------------------
    if knowledge_cache.exists():
        print(f"\n[2] Loading pre-computed store from {save_dir} (cache hit)...")
        t0 = time.perf_counter()
        store = KnowledgeStore.load(save_dir)
        ms = (time.perf_counter() - t0) * 1000.0
        print(f"    Loaded {len(store.items)} chunks in {ms:.1f}ms (no re-embed needed)")
    else:
        print("\n[2] Embedding 48 Laws into Gaussian Manifold...")
        t0 = time.perf_counter()
        store = KnowledgeStore()
        chunks: list[GaussianKnowledge] = []
        for num, title, heading, full_text in laws:
            domain = assign_domain(num)
            k = embedder.encode(
                full_text,
                item_id=f"law_{num:02d}",
                metadata={
                    "document_id": "48_laws_of_power",
                    "law_number": num,
                    "title": title,
                    "heading": heading,
                    "domain": domain,
                },
            )
            chunks.append(k)
        store.add(chunks)
        save_dir.mkdir(parents=True, exist_ok=True)
        store.save(save_dir)
        ms = (time.perf_counter() - t0) * 1000.0
        print(f"    Embedded and saved {len(chunks)} laws in {ms:.1f}ms")

    # ---- Step 3: Build Taxonomy Router ----------------------------------------
    print("\n[3] Building Hierarchical Taxonomy Router...")
    t0 = time.perf_counter()
    taxonomy = TaxonomyRouter.build_from_store(store, top_branches=5, leaf_size=8)
    ms = (time.perf_counter() - t0) * 1000.0
    bl = taxonomy.branch_labels()
    print(
        f"    {taxonomy.top_node_count} branches, {taxonomy.leaf_node_count} leaves,"
        f" {taxonomy.item_count} items in {ms:.1f}ms:"
    )
    for b in bl:
        print(f"      - [{b['id']}] {b['label']}")

    # ---- Step 4: Clef Client --------------------------------------------------
    print("\n[4] Initializing Cloudflare Workers AI Clef Client...")
    acct = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    tok  = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    if not acct or not tok:
        print("    ERROR: CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN not set in .env")
        return
    clef_client  = ClefClient(account_id=acct, api_token=tok, model="@cf/cloudflare/clef-flash")
    passage_gate = ClefPassageGate(clef_client=clef_client)
    retriever    = GaussianRetriever(store=store, taxonomy=taxonomy, passage_gate=passage_gate)
    retriever.refresh()
    print("    Clef client ready.")

    # ---- Step 5: Generator ----------------------------------------------------
    config       = load_project_config()
    provider_env = load_provider_env()
    generator    = ProviderBackedGenerator(config["generation"], provider_env)

    # ---- Queries ---------------------------------------------------------------
    test_queries = [
        (
            "How should I manage my behavior around insecure bosses, "
            "and what was the mistake of Nicolas Fouquet?",
            1,
        ),
        (
            "Why is it dangerous to hire friends in business or power, "
            "and what makes a former enemy more loyal?",
            2,
        ),
        (
            "What psychological tactic did Bobby Fischer use against "
            "Boris Spassky to break his focus?",
            17,
        ),
        (
            "How does water symbolize the ultimate defense against attacks, "
            "and why is rigidity fatal?",
            48,
        ),
    ]

    print("\n" + sep)
    print("RUNNING LIVE PIPELINE TESTS")
    print(sep)

    async with clef_client:
        for q_idx, (query_text, expected_law) in enumerate(test_queries, 1):
            print(f"\n--- [Query {q_idx}/4] ---")
            print(f"  Q: \"{query_text}\"")
            t_start = time.perf_counter()
            timings: dict[str, float] = {}

            # Stage 1: Ambiguity Gate (noul)
            t = time.perf_counter()
            ambig_resp = await clef_client.request_decision(
                state=query_text,
                questions={
                    "is_ambiguous": NoulQuestion(
                        instructions=(
                            "Does this query contain semantic ambiguity, polysemy, or multiple "
                            "distinct interpretations requiring different knowledge to answer?"
                        )
                    )
                },
            )
            ambig_ans = ambig_resp.answers.get("is_ambiguous")
            is_ambig  = bool(ambig_ans.answer) if ambig_ans else False
            p_yes = (
                ambig_ans.probabilities.get(
                    "yes",
                    float(ambig_ans.answer) if isinstance(ambig_ans.answer, (int, float)) else 0.0
                )
                if ambig_ans else 0.0
            )
            timings["disambig_ms"] = (time.perf_counter() - t) * 1000.0
            print(f"  [S1] Ambiguity Gate: ambiguous={is_ambig} (p={p_yes:.3f}) [{timings['disambig_ms']:.1f}ms]")

            # Stage 2: Taxonomy Routing (choice)
            t = time.perf_counter()
            allowed_ids, rinfo = await taxonomy.route_query_clef(
                query=query_text,
                clef_client=clef_client,
                top_n_branches=2,
            )
            timings["taxonomy_ms"] = (time.perf_counter() - t) * 1000.0
            blmap   = rinfo.get("branch_labels", {})
            pruning = rinfo.get("pruning_ratio", 0.0)
            print(f"  [S2] Taxonomy Routing: {list(blmap.values())} pruned={pruning:.1%} [{timings['taxonomy_ms']:.1f}ms]")

            # Stage 3: FTS5 Lexical Pre-filter & Native Hybrid Candidate Selection
            t = time.perf_counter()
            terms = extract_query_keywords(query_text)
            lex_hits = store.search_lexical(terms, top_k=15)
            q_rep = embedder.encode_query(query_text)
            cand_chunks = retriever.get_hybrid_candidates(
                mu=q_rep.mu,
                query_text=query_text,
                top_k=15,
                allowed_ids=allowed_ids,
            )
            cand_ids = [c.id for c in cand_chunks]
            timings["fts5_ms"] = (time.perf_counter() - t) * 1000.0
            print(f"  [S3] FTS5 Hybrid Pre-filter: {len(lex_hits)} hits -> {len(cand_ids)} candidates [{timings['fts5_ms']:.2f}ms]")

            # Stage 4: Clef Passage Gate
            t = time.perf_counter()
            gated = await passage_gate.evaluate_candidates_batch(
                query=query_text, candidates=cand_chunks, top_k=6,
            )
            survivors = [c for c, _ in gated]
            timings["gate_ms"] = (time.perf_counter() - t) * 1000.0
            print(f"  [S4] Passage Gate: {len(cand_chunks)} -> {len(survivors)} survivors [{timings['gate_ms']:.1f}ms]")

            # Stage 5: ELK Reranking
            t = time.perf_counter()
            ranked = rerank_candidates(q_rep, survivors, metric="elk")
            timings["elk_ms"] = (time.perf_counter() - t) * 1000.0
            local_total = timings["fts5_ms"] + timings["elk_ms"]
            print(f"  [S5] ELK Rerank: {len(ranked)} ranked [{timings['elk_ms']:.2f}ms] (local math: {local_total:.2f}ms)")

            if not ranked:
                print("  [WARN] No survivors; fallback to raw taxonomy candidates")
                ranked = rerank_candidates(q_rep, store.get(list(allowed_ids)[:10]), metric="elk")

            if ranked:
                top     = ranked[0].knowledge
                top_num = top.metadata.get("law_number")
                print(f"  -> Top: {top.metadata.get('heading')} (id={top.id})")
                if top_num == expected_law:
                    print(f"     [EXACT MATCH] Law {expected_law}")
                else:
                    print(f"     [RETRIEVED] Law {top_num} (expected Law {expected_law})")

            total_ms = (time.perf_counter() - t_start) * 1000.0
            print(f"  Pipeline total: {total_ms:.0f}ms")

            # Stage 6: LLM Generation (first query only)
            if q_idx == 1 and ranked:
                print("\n  [S6] Generating answer with OpenAI-compatible LLM...")
                t = time.perf_counter()
                try:
                    ans_data = generator.generate(query_text, ranked[:3])
                    gen_ms   = (time.perf_counter() - t) * 1000.0
                    print(f"  Generation: {gen_ms:.0f}ms")
                    print("\n  =================== SYNTHESIZED ANSWER ===================")
                    print("  " + ans_data["answer"].replace("\n", "\n  "))
                    print("  ==========================================================")
                except Exception as e:
                    print(f"  [WARN] LLM Generation failed ({e.__class__.__name__}: {e})")

    print("\n" + sep)
    print("ALL TESTS COMPLETED!")
    print(sep)


if __name__ == "__main__":
    asyncio.run(main())
