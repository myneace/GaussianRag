from __future__ import annotations


def uncertainty_prompt(context: str, query: str) -> str:
    return "\n".join(
        [
            "You are given retrieved knowledge chunks with confidence labels.",
            "HIGH confidence means well-established evidence.",
            "MEDIUM confidence means some ambiguity remains.",
            "LOW confidence means the evidence is contested or broad.",
            "Answer using the evidence proportionally to its confidence.",
            "",
            context,
            "",
            f"QUERY: {query}",
        ]
    )
