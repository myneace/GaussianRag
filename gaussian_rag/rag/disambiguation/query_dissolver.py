"""Agent 0: Query Dissolver — LLM-driven query-level ambiguity analysis.

This agent applies the 'Agent Dissolver' pattern to the query itself.
It analyzes the entire query for semantic complexity and projects it into
multiple high-resolution interpretations when ambiguity is detected.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Callable

from gaussian_rag.core.types import SenseNode

logger = logging.getLogger(__name__)


def _strip_markdown_json(text: str) -> str:
    """Strip leading/trailing markdown code fences from an LLM JSON response."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _stable_query_sense_id(
    term: str, domain: str, rewrite: str, version: int = 1
) -> str:
    raw = f"query::{term.lower()}::{domain.lower()}::{rewrite.lower()}::{version}"
    return f"query_dissolve_{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


LLM_QUERY_DISSOLVER_PROMPT = """
ROLE: Semantic Query Dissolver.
TASK: Analyze the input query for semantic complexity, polysemy, and latent intent.
If the query is ambiguous, project it into an exhaustive set of distinct contextual interpretations.

GUIDELINES:
- **Ambiguity Score**: Rate the query from 1 (completely literal/unambiguous) to 10 (highly polysemous/context-dependent).
- **Interpretations**: If score > 3, provide multiple distinct rewrites that clarify the different possible meanings.
- **Domain Focus**: Each interpretation should focus on a specific domain or intent.

INPUT QUERY: "{query_text}"

OUTPUT SCHEMA:
{{
  "ambiguity_score": <int 1-10>,
  "reasoning": "<short explanation>",
  "interpretations": [
    {{
      "domain": "<domain_name>",
      "text": "<clarified rewrite of the query>",
      "confidence": <float 0.0-1.0, sum must be 1.0>
    }}
  ]
}}
"""


def dissolve_query(
    text: str, llm_caller: Callable[[str], str]
) -> tuple[int, list[SenseNode]]:
    """Analyze query ambiguity and generate high-resolution interpretations.

    Returns:
        tuple of (ambiguity_score, list of SenseNodes representing interpretations)
    """
    prompt = LLM_QUERY_DISSOLVER_PROMPT.format(query_text=text)
    try:
        data = json.loads(_strip_markdown_json(llm_caller(prompt)))
        score = data.get("ambiguity_score", 1)
        interps = data.get("interpretations", [])

        nodes = []
        for item in interps:
            domain = item.get("domain", "unknown")
            rewrite = item.get("text", "")
            confidence = item.get("confidence", 1.0 / len(interps) if interps else 1.0)

            if not rewrite:
                continue

            nodes.append(
                SenseNode(
                    sense_id=_stable_query_sense_id(text[:32], domain, rewrite),
                    term=text[:32],  # Use query prefix as term
                    domain=domain,
                    confidence=confidence,
                    context_hints=[],  # Dissolver uses full text rewrite instead of hints
                    supporting_evidence=[data.get("reasoning", "")],
                    retrieval_query=rewrite,
                )
            )
        return score, nodes
    except Exception as e:
        logger.warning("Query Dissolver failed: %s", e)
        return 1, []
