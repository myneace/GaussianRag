from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AblationResult:
    name: str
    score: float
    notes: str
