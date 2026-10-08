from __future__ import annotations

import re


def split_into_chunks(
    text: str,
    *,
    max_tokens: int = 128,
    overlap: int = 16,
    min_tokens: int = 10,
) -> list[str]:
    """Split text into semantically coherent chunks for Gaussian embedding.

    Strategy (priority order):
    1. Paragraph boundaries (blank lines) are respected first — they are the
       strongest natural semantic breaks and should never be split across.
    2. Within each paragraph, sentence boundaries are used to fill chunks up to
       *max_tokens* word-tokens.
    3. An *overlap* window (in word-tokens) is carried forward so that context
       at the boundary is not lost.
    4. Orphan chunks smaller than *min_tokens* are merged into the previous
       chunk rather than being emitted as isolated, low-information nodes.

    Args:
        text:       Raw input text.
        max_tokens: Target maximum chunk size in whitespace-separated tokens.
        overlap:    Number of tokens to repeat at the start of the next chunk.
        min_tokens: Minimum tokens a chunk must have; smaller chunks are merged
                    into their predecessor.

    Returns:
        List of non-empty text chunks.
    """
    if not text or not text.strip():
        return []

    # ------------------------------------------------------------------ #
    # Step 1 — split into paragraphs (one or more blank lines)            #
    # ------------------------------------------------------------------ #
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text.strip()) if p.strip()]

    chunks: list[str] = []
    carry: list[str] = []   # overlap tokens carried from the previous chunk
    carry_tokens: int = 0

    for paragraph in paragraphs:
        # ---------------------------------------------------------- #
        # Step 2 — split paragraph into sentences                     #
        # ---------------------------------------------------------- #
        sentences = [
            seg.strip()
            for seg in re.split(r"(?<=[.!?])\s+", paragraph)
            if seg.strip()
        ]
        if not sentences:
            continue

        current: list[str] = list(carry)
        current_tokens: int = carry_tokens

        for sentence in sentences:
            sentence_tokens = len(sentence.split())

            if current and current_tokens + sentence_tokens > max_tokens:
                # -------------------------------------------------- #
                # Flush the current chunk                              #
                # -------------------------------------------------- #
                chunk_text = " ".join(current).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                # Build overlap for next chunk
                if overlap > 0:
                    all_words = " ".join(current).split()
                    overlap_words = all_words[-overlap:]
                    carry = [" ".join(overlap_words)] if overlap_words else []
                    carry_tokens = len(overlap_words)
                else:
                    carry = []
                    carry_tokens = 0

                current = list(carry)
                current_tokens = carry_tokens

            current.append(sentence)
            current_tokens += sentence_tokens

        # Flush anything left at paragraph end (no overlap carry *across*
        # paragraphs — paragraph breaks are hard semantic boundaries).
        if current:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)
        carry = []
        carry_tokens = 0

    # ------------------------------------------------------------------ #
    # Step 3 — merge orphan chunks (< min_tokens) into their predecessor  #
    # ------------------------------------------------------------------ #
    if min_tokens > 0 and len(chunks) > 1:
        merged: list[str] = [chunks[0]]
        for chunk in chunks[1:]:
            if len(chunk.split()) < min_tokens:
                merged[-1] = merged[-1] + " " + chunk
            else:
                merged.append(chunk)
        chunks = merged

    return [c for c in chunks if c.strip()]
