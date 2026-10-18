"""Leak-free train/dev/test split of skills.

Near-duplicate skills (same job, different wording or version) must land on the same side of the
split, otherwise a held-out skill is not really unseen. We group skills by description similarity
(TF-IDF cosine, union-find) and assign whole groups to a split with a fixed seed.

Run: python -m packlight.split
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
REGISTRY_PATH = DATA_DIR / "registry.json"
SPLIT_PATH = DATA_DIR / "split.json"
SIMILARITY_THRESHOLD = 0.40
SPLIT_SHARES = {"test": 0.15, "dev": 0.15}  # the rest is train
SEED = 20261018
MAX_PREFIX_FAMILY = 10  # larger prefix families (e.g. gstack) are different jobs under one brand
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to use used "
    "using when with you your user users skill skills agent agents claude code do does not".split()
)


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z][a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def is_usable(skill: dict) -> bool:
    """Drop test fixtures and bare router skills; they would only add noise to the dataset."""
    name = skill["name"].strip()
    return not name.startswith("_") and "fixture" not in skill["description"].lower() and name != "gstack"


def dedupe_by_name(skills: list[dict]) -> list[dict]:
    """Same name means same capability. Keep the version with the most content."""
    best: dict[str, dict] = {}
    for skill in skills:
        key = skill["name"].strip().lower()
        weight = skill["tokens_body"] + skill["tokens_description"]
        if key not in best or weight > best[key]["tokens_body"] + best[key]["tokens_description"]:
            best[key] = skill
    return list(best.values())


def tfidf_vectors(docs: list[list[str]]) -> list[dict[str, float]]:
    doc_freq: Counter[str] = Counter()
    for doc in docs:
        doc_freq.update(set(doc))
    total = len(docs)
    vectors = []
    for doc in docs:
        counts = Counter(doc)
        vec = {w: (1 + math.log(c)) * math.log((1 + total) / (1 + doc_freq[w])) for w, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append({w: v / norm for w, v in vec.items()})
    return vectors


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    small, large = (a, b) if len(a) < len(b) else (b, a)
    return sum(v * large.get(w, 0.0) for w, v in small.items())


def group_skills(skills: list[dict], threshold: float = SIMILARITY_THRESHOLD) -> list[list[int]]:
    docs = [tokenize(f"{s['name'].replace('-', ' ')} {s['description']}") for s in skills]
    vectors = tfidf_vectors(docs)
    parent = list(range(len(skills)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(skills)):
        for j in range(i + 1, len(skills)):
            if cosine(vectors[i], vectors[j]) >= threshold:
                parent[find(i)] = find(j)
    by_prefix: dict[str, list[int]] = {}
    for i, skill in enumerate(skills):
        prefix = skill["name"].lower().split("-")[0]
        if "-" in skill["name"]:
            by_prefix.setdefault(prefix, []).append(i)
    for members in by_prefix.values():
        if 2 <= len(members) <= MAX_PREFIX_FAMILY:
            for i in members[1:]:
                parent[find(i)] = find(members[0])
    groups: dict[int, list[int]] = {}
    for i in range(len(skills)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def assign_splits(groups: list[list[int]], total: int, seed: int = SEED) -> list[str]:
    """Whole groups go to one split. Largest groups are placed first so they cannot overshoot."""
    rng = random.Random(seed)
    order = sorted(groups, key=lambda g: (-len(g), rng.random()))
    targets = {name: round(total * share) for name, share in SPLIT_SHARES.items()}
    filled = {"test": 0, "dev": 0, "train": 0}
    result = ["train"] * total
    rng.shuffle(order)
    for group in sorted(order, key=len, reverse=True):
        split = next((n for n in ("test", "dev") if filled[n] + len(group) <= targets[n]), "train")
        filled[split] += len(group)
        for i in group:
            result[i] = split
    return result


def build() -> dict:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    skills = dedupe_by_name([c for c in registry if c["kind"] == "skill" and is_usable(c)])
    skills.sort(key=lambda s: s["name"].lower())
    groups = group_skills(skills)
    splits = ["train"] * len(skills)
    # Stratify: skills in use today and archived-only skills are split separately, so the test
    # set measures both "new skill never seen" and "skills I actually use".
    for status in ("active", "archived"):
        part = [g for g in groups if (any(skills[i]["status"] == "active" for i in g)) == (status == "active")]
        local_groups, offset = [], 0
        for g in part:
            local_groups.append(list(range(offset, offset + len(g))))
            offset += len(g)
        local = assign_splits(local_groups, offset)
        for position, i in enumerate(i for g in part for i in g):
            splits[i] = local[position]
    group_of = {i: gi for gi, g in enumerate(groups) for i in g}
    return {
        skills[i]["name"]: {"group": group_of[i], "split": splits[i], "status": skills[i]["status"]}
        for i in range(len(skills))
    }


def summarise(split: dict) -> str:
    by_split = Counter(v["split"] for v in split.values())
    group_sizes = Counter(v["group"] for v in split.values())
    biggest = group_sizes.most_common(3)
    return (
        f"{len(split)} unique skills in {len(group_sizes)} groups | "
        + ", ".join(f"{k}: {by_split[k]}" for k in ("train", "dev", "test"))
        + f" | biggest groups: {biggest}"
    )


def main() -> None:
    split = build()
    SPLIT_PATH.write_text(json.dumps(split, indent=1), encoding="utf-8")
    print(summarise(split))
    print(f"wrote {SPLIT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
