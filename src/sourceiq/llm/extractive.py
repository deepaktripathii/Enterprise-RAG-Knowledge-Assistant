"""Extractive answerer: a grounded fallback that needs no API key.

It selects the sentences from the retrieved chunks that best overlap the
question, keeps their original order, and attaches the chunk marker. The output
is less fluent than an LLM but every statement is copied from a source, which
makes it a useful offline demo mode and a safety net when the LLM API is down.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from sourceiq.llm.base import ChatMessage, LLMClient, LLMResult
from sourceiq.vectorstore.base import RetrievedChunk

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")
_BULLET = re.compile(r"^[-*•]\s+")

STOPWORDS = frozenset(
    """
    a an and are as at be been being by can could did do does for from had has have
    how i if in into is it its may me my of on or our please should so than that the
    their them then there these they this those to us was we were what when where
    which who why will with would you your
    """.split()
)


def _stem(token: str) -> str:
    """Very light plural/gerund stemming so 'passwords' matches 'password'."""
    if token.endswith("'s"):
        token = token[:-2]
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es") and token[:-2].endswith(("ch", "sh", "x", "s")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    return token


def content_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in _WORD_RE.findall(text):
        token = raw.lower()
        if len(token) < 2 or token in STOPWORDS:
            continue
        tokens.add(_stem(token))
    return tokens


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, repairing PDF line wraps first.

    PDF extraction inserts a newline at every visual line, so splitting on
    newlines would cut sentences in half ("...such as" / "medical needs."). We
    merge wrapped lines back into paragraphs, keep numbered section headings as
    separate blocks, and only then split on sentence punctuation.
    """
    blocks: list[str] = []
    current = ""
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append(current)
                current = ""
            continue
        if _HEADING.match(line) or _BULLET.match(line):
            if current:
                blocks.append(current)
            current = line
            continue
        current = f"{current} {line}".strip() if current else line
        if current.endswith(":"):
            blocks.append(current)
            current = ""
    if current:
        blocks.append(current)

    sentences: list[str] = []
    for block in blocks:
        if _HEADING.match(block) and not re.search(r"[.!?]$", block) and len(block) < 80:
            continue  # a bare section heading, not an answer sentence
        for part in _SENTENCE_SPLIT.split(block):
            part = part.strip()
            if part and len(part) > 20:
                sentences.append(part)
    return sentences


class ExtractiveLLM(LLMClient):
    provider = "extractive"
    model = "extractive-grounded-v1"

    def __init__(self, *, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def answer(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[ChatMessage] | None = None,
    ) -> LLMResult:
        question = self._question_from_prompt(user_prompt)
        question_tokens = content_tokens(question)

        candidates: list[tuple[float, int, int, str]] = []
        seen: set[str] = set()
        for rank, chunk in enumerate(chunks, start=1):
            for position, sentence in enumerate(split_sentences(chunk.text)):
                normalised = sentence.lower().strip()
                if normalised in seen:
                    continue
                overlap = question_tokens & content_tokens(sentence)
                if not overlap:
                    continue
                seen.add(normalised)
                coverage = len(overlap) / max(len(question_tokens), 1)
                score = coverage + 0.15 * chunk.score - 0.02 * rank
                candidates.append((score, rank, position, sentence, coverage))

        if not candidates:
            # Retrieval found something but no sentence shares a content word.
            fallback_text = chunks[0].text.strip().replace("\n", " ") if chunks else ""
            excerpt = fallback_text[:320].rsplit(" ", 1)[0] if fallback_text else ""
            text = (
                "The retrieved passage may be relevant, but it does not directly answer the "
                f"question. Closest excerpt: \"{excerpt} ...\" [1]"
                if excerpt
                else "I could not find this information in the available documents."
            )
            return LLMResult(text=text, model=self.model, provider=self.provider)

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score = candidates[0][0]
        # Drop tangential sentences: only keep sentences that carry a meaningful
        # share of the best match. This keeps citations focused on the passage
        # that actually answers the question.
        focused = [item for item in candidates if item[0] >= 0.6 * best_score]
        selected = sorted(focused[: self.max_sentences], key=lambda item: (item[1], item[2]))

        lines = ["Based on the retrieved documents:"]
        for _score, rank, _position, sentence, _coverage in selected:
            lines.append(f"- {sentence} [{rank}]")

        text = "\n".join(lines)
        estimated_prompt = max(1, len(user_prompt) // 4)
        estimated_completion = max(1, len(text) // 4)
        return LLMResult(
            text=text,
            model=self.model,
            provider=self.provider,
            prompt_tokens=estimated_prompt,
            completion_tokens=estimated_completion,
            metadata={"mode": "extractive"},
        )

    @staticmethod
    def _question_from_prompt(user_prompt: str) -> str:
        """Recover the question from the rendered prompt (best effort)."""
        marker = "QUESTION"
        if marker in user_prompt:
            tail = user_prompt.split(marker, 1)[1]
            tail = tail.split("\n", 1)[1] if "\n" in tail else tail
            question = tail.split("Answer using only")[0].strip()
            if question:
                return question
        return user_prompt[-400:]


__all__ = ["ExtractiveLLM", "content_tokens", "split_sentences"]
