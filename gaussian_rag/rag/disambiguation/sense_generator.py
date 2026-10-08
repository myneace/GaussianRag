"""Agent 1: Polysemy & Sense Generator — LLM-driven AI Agent.

This agent uses a Large Language Model (LLM) to dynamically analyse text and
infer polysemous terms and their specific contextual senses. It does not rely
on a hardcoded taxonomy.

It constructs a structured prompt asking the LLM to:
1. Identify tokens that carry ambiguity.
2. Deduce the possible discrete "senses" those tokens could take in different contexts.
3. Assign a confidence score based on the surrounding text.
4. Output a structured JSON response which is parsed into SenseNodes.

This makes the component framework-agnostic and infinitely flexible, adapting
to any niche domain or vocabulary present in the target text.
"""
from __future__ import annotations

import json
import hashlib
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


def _stable_sense_id(term: str, domain: str, version: int = 1) -> str:
    raw = f"{term.lower()}::{domain}::{version}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


LLM_PROMPT_TEMPLATE = """
You are a High-Resolution Semantic Sense Analyst.
Analyse the following text and identify every token or phrase that carries semantic ambiguity, polysemy, or latent multi-dimensional meaning.

TASK:
For each ambiguous term, produce an EXHAUSTIVE set of mutually exclusive senses.
- **High Resolution**: Do not limit yourself to 2 senses. If a term has 5 distinct domain interpretations relevant to this context, generate all 5.
- **Differentiable Domains**: Each sense must have unique `context_hints` (3-6 keywords) and a specific `domain` (e.g., "tech/hardware", "legal/contract", "finance/market").
- **Confidence Calibration**: Assign a 'confidence' score (0.0 to 1.0) to each sense based on the surrounding text. The sum of confidences for one term must be exactly 1.0.

Respond ONLY with valid JSON.

Example:
{{
  "terms": [
    {{
      "term": "apple",
      "senses": [
        {{
          "domain": "technology/corporate",
          "confidence": 0.8,
          "context_hints": ["iphone", "stock", "earnings"],
          "supporting_evidence": ["lexical context of finance"]
        }},
        {{
          "domain": "agriculture/botany",
          "confidence": 0.2,
          "context_hints": ["fruit", "orchard", "malus"],
          "supporting_evidence": ["alternative dictionary sense"]
        }}
      ]
    }}
  ]
}}

Text to analyse:
"{text}"
"""


def generate_senses(
    text: str,
    *,
    llm_caller: Callable[[str], str] | None = None,
    min_confidence: float = 0.2,
) -> list[SenseNode]:
    """Agent 1: LLM-driven sense generation.

    Args:
        text: Input query or passage to analyse.
        llm_caller: A function that takes a string prompt and returns an LLM's
                    string response (expected to be JSON). If None, raises ValueError.
        min_confidence: Drop senses below this normalised confidence.

    Returns:
        Flat list of SenseNodes across all detected polysemous terms.
    """
    if llm_caller is None:
        raise ValueError(
            "An LLM caller function must be provided to the AI agent to generate "
            "the taxonomy dynamically."
        )

    prompt = LLM_PROMPT_TEMPLATE.format(text=text)
    
    try:
        response_text = _strip_markdown_json(llm_caller(prompt))
        data = json.loads(response_text)
    except Exception as e:
        logger.warning("Agent 1 LLM parsing failed: %s", e)
        return []

    nodes: list[SenseNode] = []
    
    for item in data.get("terms", []):
        term = item.get("term", "").lower()
        if not term:
            continue
            
        senses = item.get("senses", [])
        if len(senses) < 2:
            continue # Need at least 2 senses to be considered polysemous
            
        for s_data in senses:
            confidence = float(s_data.get("confidence", 0.0))
            if confidence < min_confidence:
                continue
                
            domain = s_data.get("domain", "unknown")
            hints = s_data.get("context_hints", [])
            
            nodes.append(
                SenseNode(
                    sense_id=_stable_sense_id(term, domain),
                    term=term,
                    domain=domain,
                    confidence=confidence,
                    context_hints=hints,
                    supporting_evidence=s_data.get("supporting_evidence", []),
                    retrieval_query=f"{term} {' '.join(hints[:3])}".strip(),
                )
            )

    return nodes
