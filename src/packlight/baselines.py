"""Baseline scorers. Each ranks every skill for a prompt, best first. No model downloads needed."""
from __future__ import annotations

import math
from collections import Counter
from typing import Protocol

from packlight.split import tokenize

BM25_K1 = 1.5
BM25_B = 0.75
BODY_CHARS = 600


class Scorer(Protocol):
    def rank(self, prompt: str) -> list[str]: ...


def skill_text(skill: dict) -> str:
    return f"{skill['name'].replace('-', ' ')} {skill['description']} {skill['body_head'][:BODY_CHARS]}"


class Bm25:
    """Classic keyword ranking over name + description + the start of the body."""

    def __init__(self, skills: list[dict]):
        self.names = [s["name"] for s in skills]
        self.docs = [Counter(tokenize(skill_text(s))) for s in skills]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = sum(self.lengths) / max(1, len(self.docs))
        self.doc_freq: Counter[str] = Counter()
        for doc in self.docs:
            self.doc_freq.update(doc.keys())

    def scores(self, prompt: str) -> list[float]:
        total = len(self.docs)
        result = [0.0] * total
        for word in set(tokenize(prompt)):
            df = self.doc_freq.get(word, 0)
            if not df:
                continue
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            for i, doc in enumerate(self.docs):
                tf = doc.get(word, 0)
                if tf:
                    norm = tf + BM25_K1 * (1 - BM25_B + BM25_B * self.lengths[i] / self.avg_length)
                    result[i] += idf * tf * (BM25_K1 + 1) / norm
        return result

    def rank(self, prompt: str) -> list[str]:
        scores = self.scores(prompt)
        return [self.names[i] for i in sorted(range(len(scores)), key=lambda i: -scores[i])]


class TfidfCosine:
    """Cosine similarity between the prompt and each skill, with inverse-document-frequency weights."""

    def __init__(self, skills: list[dict]):
        self.names = [s["name"] for s in skills]
        docs = [tokenize(skill_text(s)) for s in skills]
        self.total = len(docs)
        self.doc_freq: Counter[str] = Counter()
        for doc in docs:
            self.doc_freq.update(set(doc))
        self.vectors = [self._vector(doc) for doc in docs]

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        vec = {
            w: (1 + math.log(c)) * math.log((1 + self.total) / (1 + self.doc_freq.get(w, 0)))
            for w, c in Counter(tokens).items()
        }
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {w: v / norm for w, v in vec.items()}

    def rank(self, prompt: str) -> list[str]:
        query = self._vector(tokenize(prompt))
        scores = [sum(v * doc.get(w, 0.0) for w, v in query.items()) for doc in self.vectors]
        return [self.names[i] for i in sorted(range(len(scores)), key=lambda i: -scores[i])]
