from __future__ import annotations


def generate_augmentations(text: str) -> list[str]:
    tokens = text.split()
    if not tokens:
        return [text]
    return [
        text,
        " ".join(tokens[: max(1, len(tokens) // 2)]),
        " ".join(tokens[max(0, len(tokens) // 3) :]),
        " ".join(reversed(tokens)),
        " ".join(dict.fromkeys(tokens)),
    ]
