"""Score a ranking method on the generated prompts.

Gold for a prompt = the skills the judge said it needs. The main set is "clean" prompts, where the
judge also confirmed the skill the prompt was written for.

Run: python -m packlight.evaluate
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from packlight.baselines import Bm25, Scorer, TfidfCosine
from packlight.dataset import PROMPTS_PATH, load_skills
from packlight.split import SPLIT_PATH

KS = (1, 3, 5)


def load_rows(path: Path = PROMPTS_PATH) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if r.get("judged")]


def is_clean(row: dict) -> bool:
    return row["written_for"] in row["needed"]


def score_prompt(ranking: list[str], gold: set[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in KS:
        top = set(ranking[:k])
        result[f"hit@{k}"] = float(bool(top & gold))
        result[f"recall@{k}"] = len(top & gold) / len(gold)
    first = next((i for i, name in enumerate(ranking, 1) if name in gold), None)
    result["mrr"] = 1 / first if first else 0.0
    return result


def evaluate(scorer: Scorer, rows: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        for key, value in score_prompt(scorer.rank(row["prompt"]), set(row["needed"])).items():
            totals[key] += value
    return {key: value / max(1, len(rows)) for key, value in totals.items()}


def format_table(results: dict[str, dict[str, dict[str, float]]], counts: dict[str, int]) -> str:
    lines = []
    for slice_name, by_method in results.items():
        lines.append(f"\n## {slice_name}  ({counts[slice_name]} prompts)")
        lines.append(f"{'method':14}" + "".join(f"{k:>10}" for k in ("hit@1", "hit@3", "hit@5", "recall@3", "mrr")))
        for method, metrics in by_method.items():
            lines.append(f"{method:14}" + "".join(f"{metrics[k]:10.3f}" for k in ("hit@1", "hit@3", "hit@5", "recall@3", "mrr")))
    return "\n".join(lines)


def main() -> None:
    skills = load_skills()
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROMPTS_PATH
    rows = [r for r in load_rows(path) if is_clean(r)]
    scorers: dict[str, Scorer] = {"bm25": Bm25(skills), "tfidf": TfidfCosine(skills)}
    slices = {
        "test skills (never seen)": [r for r in rows if split.get(r["written_for"], {}).get("split") == "test"],
        "dev skills": [r for r in rows if split.get(r["written_for"], {}).get("split") == "dev"],
        "train skills": [r for r in rows if split.get(r["written_for"], {}).get("split") == "train"],
        "active skills (in use today)": [r for r in rows if split.get(r["written_for"], {}).get("status") == "active"],
    }
    results = {name: {m: evaluate(s, subset) for m, s in scorers.items()} for name, subset in slices.items() if subset}
    print(format_table(results, {n: len(s) for n, s in slices.items()}))


if __name__ == "__main__":
    sys.exit(main())
