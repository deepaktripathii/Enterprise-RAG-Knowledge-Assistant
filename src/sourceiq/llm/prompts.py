"""Prompt construction for grounded, citation-first answers (Feature 6).

The system prompt is the first line of defence against hallucination: it forbids
outside knowledge, forces a citation marker on factual sentences, and defines an
exact refusal sentence for missing context. The context is numbered so the model
only has to emit ``[n]`` - the server maps that marker back to a real document,
page and snippet.
"""

from __future__ import annotations

from collections.abc import Sequence

from sourceiq.vectorstore.base import RetrievedChunk

SYSTEM_PROMPT = """You are SourceIQ, an enterprise knowledge assistant.

Rules:
1. Answer ONLY using the numbered CONTEXT passages provided by the user.
2. Cite every factual sentence with the matching marker, for example [1] or [2][3].
3. Never invent numbers, names, dates, policies or URLs. Do not use outside knowledge.
4. If the context does not contain the answer, reply exactly:
   "I could not find this information in the available documents."
5. Prefer short paragraphs or bullet points. Keep the answer under 180 words.
6. Quote policy wording when precision matters (limits, deadlines, eligibility).
"""

REFUSAL = "I could not find this information in the available documents."


def build_context(
    chunks: Sequence[RetrievedChunk],
    *,
    max_chars: int = 8000,
) -> str:
    """Render retrieved chunks as numbered context blocks."""
    blocks: list[str] = []
    used = 0
    for index, chunk in enumerate(chunks, start=1):
        header = (
            f"[{index}] Source: {chunk.filename} | page {chunk.page} | "
            f"relevance {chunk.score:.2f}"
        )
        budget = max_chars - used - len(header) - 2
        if budget <= 80:
            break
        text = chunk.text.strip()
        if len(text) > budget:
            text = text[:budget].rsplit(" ", 1)[0] + " ..."
        block = f"{header}\n{text}"
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks)


def build_user_prompt(question: str, context: str) -> str:
    return (
        "CONTEXT\n"
        "-------\n"
        f"{context}\n\n"
        "QUESTION\n"
        "--------\n"
        f"{question}\n\n"
        "Answer using only the context above and cite your sources with [n]."
    )


__all__ = ["REFUSAL", "SYSTEM_PROMPT", "build_context", "build_user_prompt"]
