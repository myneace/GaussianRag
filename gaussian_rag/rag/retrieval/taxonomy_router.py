"""taxonomy_router.py — Hierarchical knowledge taxonomy for branch-pruning candidate generation.

Ported and adapted from jev-rag/taxonomy.py for native integration with GaussianRag.

Key changes from the jev-rag version:
- Operates on GaussianKnowledge chunks (using mu vectors as embeddings).
- `build_from_store()` classmethod constructs a TaxonomyRouter directly from a KnowledgeStore.
- `route_query()` returns a set of allowed chunk IDs for pre-filtering the ANN search.
- Persists taxonomy index alongside knowledge.json / lexical.db as taxonomy.json + taxonomy.npz.
- Integrates Clef-based branch routing as an optional override when a ClefClient is available.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants (mirrored from jev-rag/taxonomy.py with Gaussian-specific defaults)
# ──────────────────────────────────────────────────────────────────────────────
TAXONOMY_VERSION = 1
DEFAULT_TOP_BRANCHES = 12
DEFAULT_LEAF_SIZE = 64
DEFAULT_ROUTE_LEAVES = 4
DEFAULT_ROUTE_MIN_ITEMS = 50   # Lowered from 200 — GaussianRag stores are typically smaller
DEFAULT_SECONDARY_MARGIN = 0.025
DEFAULT_TAXONOMY_RRF_K = 60
DEFAULT_TAXONOMY_ROUTE_WEIGHT = 2.0

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u3400-\u4dbf\u4e00-\u9fff]{2,8}")
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
    "has", "have", "had", "not", "but", "can", "may", "into", "than", "then",
    "their", "there", "these", "those", "using", "used", "use", "between", "about",
    "what", "when", "where", "which", "who", "why", "how", "also", "more", "such",
    "study", "studies", "result", "results", "effect", "effects", "patients", "patient",
    "based", "data", "analysis", "associated", "compared", "however", "during", "after",
    "before", "over", "under", "through", "other", "some", "each", "both", "all",
}


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers (pure numpy, no sklearn dependency)
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("taxonomy vectors must be a two-dimensional matrix")
    if not len(matrix):
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _spherical_kmeans(
    vectors: np.ndarray, cluster_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic spherical k-means with no sklearn dependency."""
    vectors = _normalise_rows(vectors)
    count = len(vectors)
    if not count:
        return np.empty(0, dtype=np.int32), np.empty((0, 0), dtype=np.float32)
    cluster_count = max(1, min(int(cluster_count), count))
    if cluster_count == 1:
        center = vectors.mean(axis=0, keepdims=True)
        center = _normalise_rows(center)
        return np.zeros(count, dtype=np.int32), center

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, count))
    selected = [first]
    closest = vectors @ vectors[first]
    for _ in range(1, cluster_count):
        next_index = int(np.argmin(closest))
        selected.append(next_index)
        closest = np.maximum(closest, vectors @ vectors[next_index])
    centers = vectors[np.asarray(selected)].copy()
    assignments = np.full(count, -1, dtype=np.int32)

    for _ in range(30):
        similarities = vectors @ centers.T
        updated = np.argmax(similarities, axis=1).astype(np.int32)
        if np.array_equal(updated, assignments):
            break
        assignments = updated
        new_centers = []
        best_similarity = similarities.max(axis=1)
        for cluster in range(cluster_count):
            members = vectors[assignments == cluster]
            if len(members):
                center = members.mean(axis=0)
            else:
                replacement = int(np.argmin(best_similarity))
                center = vectors[replacement]
                assignments[replacement] = cluster
            center = center / max(float(np.linalg.norm(center)), 1e-12)
            new_centers.append(center)
        centers = np.asarray(new_centers, dtype=np.float32)
    return assignments, centers


def _representative_terms(texts: list[str], limit: int = 5) -> list[str]:
    documents: list[list[str]] = []
    frequencies: Counter[str] = Counter()
    document_frequency: Counter[str] = Counter()
    for text in texts:
        tokens = [
            token.lower()
            for token in _WORD_RE.findall(text[:2400])
            if token.lower() not in _STOPWORDS and not token.isdigit()
        ]
        if not tokens:
            continue
        documents.append(tokens)
        frequencies.update(tokens)
        document_frequency.update(set(tokens))
    total = max(1, len(documents))
    scored = {
        token: frequencies[token] * math.log((total + 1) / (document_frequency[token] + 1) + 1)
        for token in frequencies
    }
    return sorted(scored, key=lambda t: (-scored[t], t))[:limit]


def _node_label(texts: list[str], fallback: str) -> tuple[str, str]:
    terms = _representative_terms(texts)
    label = " / ".join(terms[:3]) if terms else fallback
    description = ", ".join(terms) if terms else fallback
    return label, description


# ──────────────────────────────────────────────────────────────────────────────
# Core taxonomy builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_taxonomy_tree(
    vectors: np.ndarray,
    item_ids: list[str],
    texts: list[str],
    *,
    top_branches: int = DEFAULT_TOP_BRANCHES,
    leaf_size: int = DEFAULT_LEAF_SIZE,
    secondary_margin: float = DEFAULT_SECONDARY_MARGIN,
    seed: int = 17,
) -> tuple[dict[str, Any], np.ndarray]:
    """Build a two-level category tree and multi-label leaf assignments."""
    vectors = _normalise_rows(vectors)
    if len(item_ids) != len(vectors) or len(texts) != len(vectors):
        raise ValueError("item_ids, texts, and vectors must have the same length")
    if not item_ids:
        raise ValueError("cannot build a taxonomy for an empty corpus")
    if leaf_size < 2:
        raise ValueError("leaf_size must be at least 2")

    top_count = max(1, min(top_branches, math.ceil(len(item_ids) / leaf_size)))
    top_assignments, top_centroids = _spherical_kmeans(vectors, top_count, seed)
    nodes: list[dict[str, Any]] = [
        {
            "id": "root",
            "parent_id": None,
            "depth": 0,
            "label": "All knowledge",
            "description": "Root of the automatically generated knowledge taxonomy",
            "item_count": len(item_ids),
        }
    ]
    leaf_centroids: list[np.ndarray] = []
    leaf_parent_ids: list[str] = []
    primary_leaf_indexes = np.full(len(item_ids), -1, dtype=np.int32)

    for top_index in range(top_count):
        member_indexes = np.flatnonzero(top_assignments == top_index)
        top_id = f"topic-{top_index + 1:02d}"
        top_texts = [texts[int(i)] for i in member_indexes]
        label, description = _node_label(top_texts, top_id)
        nodes.append(
            {
                "id": top_id,
                "parent_id": "root",
                "depth": 1,
                "label": label,
                "description": description,
                "item_count": int(len(member_indexes)),
            }
        )
        local_count = max(1, math.ceil(len(member_indexes) / leaf_size))
        local_assignments, local_centroids = _spherical_kmeans(
            vectors[member_indexes], local_count, seed + 1009 * (top_index + 1)
        )
        for local_index in range(local_count):
            local_members = member_indexes[local_assignments == local_index]
            leaf_id = f"{top_id}/leaf-{local_index + 1:02d}"
            leaf_texts = [texts[int(i)] for i in local_members]
            leaf_label, leaf_description = _node_label(leaf_texts, leaf_id)
            nodes.append(
                {
                    "id": leaf_id,
                    "parent_id": top_id,
                    "depth": 2,
                    "label": leaf_label,
                    "description": leaf_description,
                    "item_count": int(len(local_members)),
                }
            )
            global_leaf_index = len(leaf_centroids)
            leaf_centroids.append(local_centroids[local_index])
            leaf_parent_ids.append(top_id)
            primary_leaf_indexes[local_members] = global_leaf_index

    centroid_matrix = _normalise_rows(np.asarray(leaf_centroids, dtype=np.float32))
    similarities = vectors @ centroid_matrix.T
    assignments: dict[str, list[str]] = {}
    leaf_ids = [node["id"] for node in nodes if node["depth"] == 2]
    for item_index, item_id in enumerate(item_ids):
        primary = int(primary_leaf_indexes[item_index])
        primary_score = float(similarities[item_index, primary])
        ordered = np.argsort(-similarities[item_index], kind="stable")
        memberships = [leaf_ids[primary]]
        for candidate in ordered:
            candidate = int(candidate)
            if candidate == primary:
                continue
            if float(similarities[item_index, candidate]) < primary_score - secondary_margin:
                break
            memberships.append(leaf_ids[candidate])
            break
        assignments[str(item_id)] = memberships

    node_by_id = {node["id"]: node for node in nodes}
    for node in nodes:
        if node["depth"] != 2:
            continue
        node["assigned_count"] = sum(
            node["id"] in membership for membership in assignments.values()
        )

    tree = {
        "version": TAXONOMY_VERSION,
        "config": {
            "top_branches": top_branches,
            "leaf_size": leaf_size,
            "secondary_margin": secondary_margin,
            "seed": seed,
        },
        "item_count": len(item_ids),
        "top_node_count": top_count,
        "leaf_node_count": len(leaf_ids),
        "nodes": nodes,
        "assignments": assignments,
        "leaf_ids": leaf_ids,
        "leaf_parent_ids": leaf_parent_ids,
        "labels": {nid: node_by_id[nid]["label"] for nid in node_by_id},
    }
    return tree, centroid_matrix


# ──────────────────────────────────────────────────────────────────────────────
# TaxonomyRouter — the public interface for Phase 4
# ──────────────────────────────────────────────────────────────────────────────

class TaxonomyRouter:
    """Hierarchical branch-pruning router for GaussianRetriever candidate pre-filtering.

    Build with `TaxonomyRouter.build_from_store(store)` or load from disk with
    `TaxonomyRouter.load(store_directory)`, or instantiate an empty router with
    `TaxonomyRouter()` or `TaxonomyRouter(store)`.

    Once built, call `route_query(query_mu)` to get a set of allowed chunk IDs
    that can be used to filter the ANN candidate pool before manifold reranking.

    Persistence:
        - `taxonomy.json`  — tree structure, node labels, and chunk assignments.
        - `taxonomy.npz`   — leaf centroid matrix (float32).
    """

    def __init__(
        self,
        tree_or_store: Any = None,
        centroids: np.ndarray | None = None,
        *,
        _tree: dict[str, Any] | None = None,
        _centroids: np.ndarray | None = None,
        store: Any = None,
        tree: dict[str, Any] | None = None,
        top_branches: int = DEFAULT_TOP_BRANCHES,
        leaf_size: int = DEFAULT_LEAF_SIZE,
        secondary_margin: float = DEFAULT_SECONDARY_MARGIN,
        seed: int = 17,
    ) -> None:
        target_store = store
        if tree_or_store is not None and hasattr(tree_or_store, "values"):
            target_store = tree_or_store

        if target_store is not None:
            chunks = target_store.values()
            if chunks:
                built = self.build_from_store(
                    target_store,
                    top_branches=top_branches,
                    leaf_size=leaf_size,
                    secondary_margin=secondary_margin,
                    seed=seed,
                )
                self._tree = built._tree
                self._centroids = built._centroids
                return

        resolved_tree = (
            _tree
            if _tree is not None
            else (tree if tree is not None else (tree_or_store if isinstance(tree_or_store, dict) else {}))
        )
        resolved_centroids = (
            _centroids
            if _centroids is not None
            else (centroids if centroids is not None else np.empty((0, 0), dtype=np.float32))
        )
        self._tree = resolved_tree
        self._centroids = np.asarray(resolved_centroids, dtype=np.float32)

    @property
    def is_empty(self) -> bool:
        """Return True if the taxonomy index is empty and unbuilt."""
        return (
            len(self._centroids) == 0
            or not self._tree
            or not self._tree.get("leaf_ids")
        )

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def build_from_store(
        cls,
        store: Any,  # KnowledgeStore — avoid circular import
        *,
        top_branches: int = DEFAULT_TOP_BRANCHES,
        leaf_size: int = DEFAULT_LEAF_SIZE,
        secondary_margin: float = DEFAULT_SECONDARY_MARGIN,
        seed: int = 17,
    ) -> "TaxonomyRouter":
        """Build taxonomy index from all chunks in a KnowledgeStore.

        Uses the mean embedding vector (mu) of each GaussianKnowledge chunk as
        the feature vector for spherical k-means clustering.

        Args:
            store: KnowledgeStore instance with ingested chunks.
            top_branches: Target number of top-level topic branches.
            leaf_size: Target number of items per leaf cluster.
            secondary_margin: Cosine similarity margin for secondary memberships.
            seed: Random seed for deterministic clustering.

        Returns:
            Constructed TaxonomyRouter ready for routing queries.
        """
        chunks = store.values()
        if not chunks:
            raise ValueError("Cannot build taxonomy from an empty KnowledgeStore")

        item_ids = [chunk.id for chunk in chunks]
        texts = []
        for chunk in chunks:
            prefix = ""
            if chunk.metadata:
                title = chunk.metadata.get("title") or chunk.metadata.get("document_id") or chunk.metadata.get("heading")
                if title:
                    prefix = f"{title} {title} "
            texts.append(f"{prefix}{chunk.text}")
        vectors = np.stack([chunk.mu.astype(np.float32) for chunk in chunks])

        tree, centroids = _build_taxonomy_tree(
            vectors,
            item_ids,
            texts,
            top_branches=top_branches,
            leaf_size=leaf_size,
            secondary_margin=secondary_margin,
            seed=seed,
        )
        logger.info(
            "Taxonomy built: %d items → %d top branches, %d leaf clusters",
            len(item_ids),
            tree["top_node_count"],
            tree["leaf_node_count"],
        )
        return cls(_tree=tree, _centroids=centroids)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        """Persist taxonomy alongside knowledge.json / lexical.db."""
        if self.is_empty:
            logger.debug("TaxonomyRouter is empty; skipping save.")
            return
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "taxonomy.json").write_text(
            json.dumps(self._tree, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with (path / "taxonomy.npz").open("wb") as handle:
            np.savez(handle, leaf_centroids=np.asarray(self._centroids, dtype=np.float32))
        logger.info("TaxonomyRouter saved to %s", path)

    @classmethod
    def load(cls, directory: str | Path) -> "TaxonomyRouter | None":
        """Load a previously persisted taxonomy from a store directory.

        Returns None if no valid taxonomy index exists (graceful degradation).
        """
        path = Path(directory)
        json_path = path / "taxonomy.json"
        npz_path = path / "taxonomy.npz"
        if not json_path.exists() or not npz_path.exists():
            return None
        tree = json.loads(json_path.read_text(encoding="utf-8"))
        if int(tree.get("version", -1)) != TAXONOMY_VERSION:
            logger.warning("Taxonomy version mismatch at %s — rebuilding required", path)
            return None
        with np.load(npz_path, allow_pickle=False) as cached:
            centroids = cached["leaf_centroids"].copy()
        if len(centroids) != int(tree.get("leaf_node_count", -1)):
            logger.warning("Taxonomy centroid count mismatch at %s — rebuilding required", path)
            return None
        logger.info("TaxonomyRouter loaded from %s (%d leaves)", path, len(centroids))
        return cls(_tree=tree, _centroids=centroids)

    # ── Query routing ─────────────────────────────────────────────────────────

    def route_query(
        self,
        query_mu: np.ndarray,
        *,
        route_leaves: int = DEFAULT_ROUTE_LEAVES,
        min_items: int = DEFAULT_ROUTE_MIN_ITEMS,
    ) -> tuple[set[str], dict[str, Any]]:
        """Select query-relevant leaves, expanding until a safe corpus scope exists.

        Args:
            query_mu: Mean vector of the query Gaussian (shape [d]).
            route_leaves: Minimum number of leaf clusters to activate.
            min_items: Minimum total allowed chunk IDs before stopping expansion.

        Returns:
            (allowed_ids, routing_info) where allowed_ids is a set of chunk ID
            strings from the activated leaf clusters, and routing_info contains
            diagnostic metadata (routes, scope_items, tree statistics).
        """
        if self.is_empty:
            return set(), {
                "routes": [],
                "selected_leaf_count": 0,
                "scope_items": 0,
                "tree_items": 0,
                "top_node_count": 0,
                "leaf_node_count": 0,
                "status": "empty",
            }

        vector = np.asarray(query_mu, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            vector = vector / norm

        centroids = _normalise_rows(self._centroids)
        similarities = centroids @ vector
        ordered = np.argsort(-similarities, kind="stable").tolist()

        tree = self._tree
        assignments = tree["assignments"]
        leaf_ids = tree["leaf_ids"]
        labels = tree.get("labels") or {}

        selected: list[int] = []
        allowed: set[str] = set()
        for leaf_index in ordered:
            leaf_index = int(leaf_index)
            selected.append(leaf_index)
            leaf_id = leaf_ids[leaf_index]
            allowed.update(
                item_id
                for item_id, memberships in assignments.items()
                if leaf_id in memberships
            )
            if len(selected) >= max(1, route_leaves) and len(allowed) >= max(1, min_items):
                break

        routes = [
            {
                "id": leaf_ids[idx],
                "label": labels.get(leaf_ids[idx], leaf_ids[idx]),
                "parent_id": tree["leaf_parent_ids"][idx],
                "score": round(float(similarities[idx]), 6),
            }
            for idx in selected
        ]
        routing_info: dict[str, Any] = {
            "routes": routes,
            "selected_leaf_count": len(selected),
            "scope_items": len(allowed),
            "tree_items": int(tree["item_count"]),
            "top_node_count": int(tree["top_node_count"]),
            "leaf_node_count": int(tree["leaf_node_count"]),
        }
        return allowed, routing_info

    # ── Diagnostic helpers ────────────────────────────────────────────────────

    @property
    def item_count(self) -> int:
        return int(self._tree.get("item_count", 0))

    @property
    def leaf_node_count(self) -> int:
        return int(self._tree.get("leaf_node_count", 0))

    @property
    def top_node_count(self) -> int:
        return int(self._tree.get("top_node_count", 0))

    def branch_labels(self) -> list[dict[str, str]]:
        """Return top-level branch labels for logging / inspection."""
        return [
            {"id": node["id"], "label": node["label"]}
            for node in self._tree.get("nodes", [])
            if node["depth"] == 1
        ]

    async def route_query_clef(
        self,
        query: str,
        clef_client: Any,
        *,
        top_n_branches: int = 2,
        min_confidence: float = 0.05,
    ) -> tuple[set[str], dict[str, Any]]:
        """Route query to top taxonomy branches via real Cloudflare Workers AI Clef choice question.

        Args:
            query: User search query string.
            clef_client: ClefClient instance with real Cloudflare credentials.
            top_n_branches: Maximum number of branches to activate.
            min_confidence: Minimum probability threshold for secondary branch inclusion.

        Returns:
            (allowed_ids, routing_info) containing the set of chunk IDs in the selected branches.
        """
        if self.is_empty:
            return set(), {
                "selected_branches": [],
                "primary_branch": "",
                "branch_labels": {},
                "probabilities": {},
                "elapsed_ms": 0.0,
                "scope_items": 0,
                "tree_items": 0,
                "pruning_ratio": 0.0,
                "status": "empty",
            }

        top_nodes = [node for node in self._tree.get("nodes", []) if node["depth"] == 1]
        if not top_nodes:
            raise ValueError("Taxonomy tree contains no top-level branches for routing")

        criteria = {}
        for node in top_nodes:
            label = node["label"]
            desc = node.get("description", "")
            criteria[node["id"]] = f"Branch covering {label}. Concepts: {desc}".strip()

        from gaussian_rag.rag.disambiguation.clef_client import ChoiceQuestion

        question = ChoiceQuestion(
            instructions=(
                "You are an expert document taxonomy router. Select the topic branch that best categorizes "
                "the subject matter and contains relevant evidence to answer the user query."
            ),
            criteria=criteria,
        )

        response = await clef_client.request_decision(
            state=f"Query to route: {query}",
            questions={"taxonomy_branch": question},
        )

        answer = response.answers.get("taxonomy_branch")
        if not answer:
            raise RuntimeError(f"Clef-Flash returned no answer for taxonomy_branch: {response}")

        selected_primary = str(answer.answer)
        probs = answer.probabilities or {}

        # Rank branches: primary answer first, followed by others sorted by probability
        ranked_branches = [selected_primary] if selected_primary in criteria else []
        sorted_by_prob = sorted(
            [b for b in probs.keys() if b != selected_primary and b in criteria],
            key=lambda b: probs.get(b, 0.0),
            reverse=True,
        )
        for b in sorted_by_prob:
            if len(ranked_branches) >= top_n_branches:
                break
            if probs.get(b, 0.0) >= min_confidence:
                ranked_branches.append(b)

        if not ranked_branches and top_nodes:
            ranked_branches = [top_nodes[0]["id"]]

        # Collect allowed chunk IDs belonging to selected branches
        selected_set = set(ranked_branches)
        leaf_parent_map = dict(zip(self._tree["leaf_ids"], self._tree["leaf_parent_ids"]))

        allowed_ids: set[str] = set()
        for item_id, memberships in self._tree["assignments"].items():
            for lid in memberships:
                parent = leaf_parent_map.get(lid)
                if parent in selected_set or any(lid.startswith(f"{b}/") for b in selected_set):
                    allowed_ids.add(item_id)
                    break

        pruning_ratio = 1.0 - (len(allowed_ids) / max(self.item_count, 1))

        routing_info: dict[str, Any] = {
            "selected_branches": ranked_branches,
            "primary_branch": selected_primary,
            "branch_labels": {b: self._tree["labels"].get(b, b) for b in ranked_branches},
            "probabilities": probs,
            "elapsed_ms": response.elapsed_ms,
            "scope_items": len(allowed_ids),
            "tree_items": self.item_count,
            "pruning_ratio": round(pruning_ratio, 4),
        }
        return allowed_ids, routing_info

