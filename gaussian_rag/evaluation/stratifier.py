from __future__ import annotations

import math
from collections import Counter


def lexical_entropy(text: str) -> float:
    """Shannon entropy (bits) of unigram token distribution.

    Returns 0.0 (not -0.0) for degenerate single-token inputs.
    """
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    # Guard against -0.0 (occurs when all tokens are identical, log2(1) == 0)
    return entropy + 0.0


def ambiguity_bucket(text: str) -> str:
    entropy = lexical_entropy(text)
    if entropy >= 4.0:
        return "high"
    if entropy >= 2.5:
        return "medium"
    return "low"
