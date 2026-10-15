from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


class BenchmarkSample(TypedDict):
    query: str
    relevant_ids: list[str]


def load_json_benchmark(path: str | Path) -> list[BenchmarkSample]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples: list[BenchmarkSample] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        query = item.get("query")
        relevant_ids = item.get("relevant_ids")
        if not isinstance(query, str) or not isinstance(relevant_ids, list):
            continue
        normalized_ids = [value for value in relevant_ids if isinstance(value, str)]
        samples.append(BenchmarkSample(query=query, relevant_ids=normalized_ids))
    return samples


def demo_benchmark() -> list[BenchmarkSample]:
    return [
        BenchmarkSample(query="treatment for mitral valve issues", relevant_ids=["medical-1-chunk-0", "medical-3-chunk-0"]),
        BenchmarkSample(query="ambiguity in legal systems", relevant_ids=["legal-1-chunk-0"]),
        BenchmarkSample(query="Apple stock and market competition", relevant_ids=["tech-1-chunk-0", "tech-3-chunk-0"]),
        BenchmarkSample(query="best season for apple harvest", relevant_ids=["nature-3-chunk-0"]),
        BenchmarkSample(query="nutritional value of crisp apples", relevant_ids=["nature-2-chunk-0"]),
        BenchmarkSample(query="central bank interest rate policy", relevant_ids=["finance-2-chunk-0"]),
        BenchmarkSample(query="erosion along the river bank", relevant_ids=["geography-1-chunk-0"]),
        BenchmarkSample(query="fishing spots on the grassy bank", relevant_ids=["geography-2-chunk-0", "geography-3-chunk-0"]),
        BenchmarkSample(query="clean code in python programming", relevant_ids=["coding-1-chunk-0", "coding-3-chunk-0"]),
        BenchmarkSample(query="largest constricting snakes in Asia", relevant_ids=["snakes-1-chunk-0", "snakes-3-chunk-0"]),
        BenchmarkSample(query="how pythons hunt their prey", relevant_ids=["snakes-2-chunk-0"]),
    ]
