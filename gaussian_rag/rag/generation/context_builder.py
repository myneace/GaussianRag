from __future__ import annotations

from gaussian_rag.core.types import RetrievedChunk


def build_context(query: str, retrieved: list[RetrievedChunk]) -> str:
    # Map uncertainty labels to confidence levels for the LLM prompt
    # Uncertainty: LOW    -> Confidence: HIGH
    # Uncertainty: MEDIUM -> Confidence: MEDIUM
    # Uncertainty: HIGH   -> Confidence: LOW
    confidence_map = {
        "low": "HIGH",
        "medium": "MEDIUM",
        "high": "LOW",
    }

    lines = ["[CONTEXT START]", "", f"[QUERY] {query}", ""]
    for index, item in enumerate(retrieved, start=1):
        conf_label = confidence_map.get(item.uncertainty_label.lower(), "MEDIUM")
        lines.extend(
            [
                f"[Chunk {index}] — Confidence: {conf_label}",
                item.knowledge.text,
                "",
            ]
        )
    lines.append("[CONTEXT END]")
    return "\n".join(lines)


def build_multi_sense_context(
    query: str,
    sense_results: dict[str, list[RetrievedChunk]],
    *,
    sense_domains: dict[str, str] | None = None,
    max_total_chunks: int | None = None,
) -> str:
    """Build a multi-sense LLM prompt context with explicit domain labels.

    Args:
        query: The original user query.
        sense_results: Mapping of sense_id -> ranked chunks for that sense.
                       sense_id should carry the domain in its metadata.
    """
    confidence_map = {"low": "HIGH", "medium": "MEDIUM", "high": "LOW"}

    lines = ["[CONTEXT START — MULTI-SENSE]", "", f"[QUERY] {query}", ""]

    sense_labels = [chr(ord("A") + i) for i in range(len(sense_results))]
    seen_chunk_ids: set[str] = set()
    emitted_chunks = 0

    for label, (sense_id, chunks) in zip(sense_labels, sense_results.items()):
        domain = sense_id
        if sense_domains is not None:
            domain = sense_domains.get(sense_id, sense_id)
        elif chunks:
            domain = chunks[0].knowledge.metadata.get("domain", sense_id)
        lines.append(f"[Sense {label}: {domain}]")
        if not chunks or (
            max_total_chunks is not None and emitted_chunks >= max_total_chunks
        ):
            lines.append("  (no results for this sense)")
            lines.append("")
            continue
        displayed = 0
        for index, item in enumerate(chunks, start=1):
            chunk_id = item.knowledge.id
            if chunk_id in seen_chunk_ids:
                continue
            if max_total_chunks is not None and emitted_chunks >= max_total_chunks:
                break
            conf_label = confidence_map.get(item.uncertainty_label.lower(), "MEDIUM")
            lines.extend(
                [
                    f"  [Chunk {label}{index}] — Confidence: {conf_label}",
                    f"  {item.knowledge.text}",
                    "",
                ]
            )
            seen_chunk_ids.add(chunk_id)
            emitted_chunks += 1
            displayed += 1
        if displayed == 0:
            lines.append("  (no unique results for this sense within top-k budget)")
            lines.append("")

    lines.append("[CONTEXT END]")
    return "\n".join(lines)
