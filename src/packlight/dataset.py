"""Generate and check training/eval prompts for each skill.

Step 1 (generate): ask a free model for realistic user messages that need a given skill, in
several styles, then drop any that leak the skill's own name.
Step 2 (judge): for each prompt, ask a model which of the likely candidate skills it needs. The
gold label is the set the judge says are needed, not only the skill it was written for, because
sibling skills often apply to the same prompt.

Run a small sample: python -m packlight.dataset sample 6
"""
from __future__ import annotations

import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from packlight.llm import Gateway
from packlight.split import SEED, REGISTRY_PATH, SPLIT_PATH, cosine, tfidf_vectors, tokenize

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROMPTS_PATH = DATA_DIR / "prompts.jsonl"
PROMPTS_PER_SKILL = 12
JUDGE_CANDIDATES = 5
WORKERS = 3
BODY_CHARS = 600

GEN_SYSTEM = (
    "You write realistic messages that a software developer types to an AI coding agent. "
    "Reply with a JSON array of strings only."
)
GEN_TEMPLATE = """Skill the agent has available:
name: {name}
description: {description}
start of its instructions: {body}

Write {n} different user messages where this skill would clearly be needed. Mix these styles:
- 3 direct requests that state the task plainly
- 3 indirect ones that describe the situation or goal without naming the skill or its jargon
- 3 short, terse, lowercase messages with a typo or two, the way someone types fast
- 3 longer ones with context (a paragraph, a file or error mentioned)
Rules: never write the skill name "{name}". Do not mention that a skill exists. Vary topics and
wording. Return only a JSON array of {n} strings."""

JUDGE_SYSTEM = "You judge which tools an AI agent would need. Reply with a JSON array of ids only."
JUDGE_TEMPLATE = """User message:
{prompt}

Candidate skills:
{candidates}

Which candidates would a good agent actually need to load to do this well? Be strict: only
include a skill if the message clearly needs what it provides. An empty array is allowed.
Return only a JSON array of the ids (numbers)."""


def parse_json_array(text: str) -> list:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def clean_prompts(items: list) -> list[str]:
    seen: set[str] = set()
    result = []
    for item in items:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if len(text) < 4 or text.lower() in seen:
            continue
        seen.add(text.lower())
        result.append(text)
    return result


def leaks_name(prompt: str, skill_name: str) -> bool:
    lowered = prompt.lower()
    words = re.split(r"[-_\s]+", skill_name.lower().strip("_"))
    if len(words) == 1:
        return re.search(rf"\b{re.escape(words[0])}\b", lowered) is not None
    return any(joiner.join(words) in lowered for joiner in ("-", " ", "_", ""))


def generate_for_skill(gateway: Gateway, skill: dict, n: int = PROMPTS_PER_SKILL) -> list[str]:
    message = GEN_TEMPLATE.format(
        name=skill["name"], description=skill["description"],
        body=skill["body_head"][:BODY_CHARS].replace("\n", " "), n=n,
    )
    reply = gateway.chat([{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": message}])
    return [p for p in clean_prompts(parse_json_array(reply)) if not leaks_name(p, skill["name"])]


def nearest_siblings(skills: list[dict], index: int, count: int, vectors: list[dict]) -> list[int]:
    scored = sorted(
        (j for j in range(len(skills)) if j != index),
        key=lambda j: cosine(vectors[index], vectors[j]), reverse=True,
    )
    return scored[:count]


def judge_prompt(gateway: Gateway, prompt: str, candidates: list[dict]) -> list[str]:
    listing = "\n".join(f"{i}. {c['name']}: {c['description'][:200]}" for i, c in enumerate(candidates))
    reply = gateway.chat(
        [{"role": "system", "content": JUDGE_SYSTEM},
         {"role": "user", "content": JUDGE_TEMPLATE.format(prompt=prompt, candidates=listing)}],
        temperature=0.0, max_tokens=200,
    )
    ids = [i for i in parse_json_array(reply) if isinstance(i, int) and 0 <= i < len(candidates)]
    return [candidates[i]["name"] for i in ids]


def load_skills() -> list[dict]:
    from packlight.split import dedupe_by_name, is_usable
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    skills = dedupe_by_name([c for c in registry if c["kind"] == "skill" and is_usable(c)])
    skills.sort(key=lambda s: s["name"].lower())
    return skills


def process_skill(gateway: Gateway, skills: list[dict], vectors: list[dict], index: int) -> list[dict]:
    skill = skills[index]
    rows = []
    for prompt in generate_for_skill(gateway, skill):
        siblings = nearest_siblings(skills, index, JUDGE_CANDIDATES, vectors)
        candidates = [skill] + [skills[j] for j in siblings]
        random.Random(f"{SEED}:{prompt}").shuffle(candidates)
        needed = judge_prompt(gateway, prompt, candidates)
        rows.append({"prompt": prompt, "written_for": skill["name"], "needed": needed,
                     "candidates": [c["name"] for c in candidates]})
    return rows


def run(indices: list[int], out_path: Path = PROMPTS_PATH) -> None:
    skills = load_skills()
    vectors = tfidf_vectors([tokenize(f"{s['name'].replace('-', ' ')} {s['description']}") for s in skills])
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    done = set()
    if out_path.exists():
        done = {json.loads(line)["written_for"] for line in out_path.read_text(encoding="utf-8").splitlines() if line}
    todo = [i for i in indices if skills[i]["name"] not in done]
    gateway = Gateway()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(WORKERS) as pool:
        for rows in pool.map(lambda i: process_skill(gateway, skills, vectors, i), todo):
            for row in rows:
                row["split"] = split.get(row["written_for"], {}).get("split", "train")
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{rows[0]['written_for'] if rows else '(none)'}: {len(rows)} prompts", flush=True)


def main(argv: list[str]) -> None:
    if argv[:1] == ["sample"]:
        count = int(argv[1]) if len(argv) > 1 else 6
        indices = random.Random(SEED).sample(range(len(load_skills())), count)
        run(indices, DATA_DIR / "prompts_sample.jsonl")
    elif argv[:1] == ["all"]:
        run(list(range(len(load_skills()))))
    else:
        print("usage: python -m packlight.dataset sample [N] | all")


if __name__ == "__main__":
    main(sys.argv[1:])
