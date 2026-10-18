import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from packlight.baselines import Bm25, TfidfCosine
from packlight.evaluate import evaluate, score_prompt

SKILLS = [
    {"name": "browser-testing", "description": "Automate a web browser to test pages and click buttons", "body_head": ""},
    {"name": "sql-migrations", "description": "Write reversible database schema migrations for postgres", "body_head": ""},
    {"name": "poster-design", "description": "Design posters and print layouts with strong typography", "body_head": ""},
]


def test_both_baselines_put_the_obvious_skill_first():
    for scorer in (Bm25(SKILLS), TfidfCosine(SKILLS)):
        assert scorer.rank("add a migration for the postgres schema")[0] == "sql-migrations"
        assert scorer.rank("click the button and test the web page")[0] == "browser-testing"


def test_score_prompt_hit_recall_and_mrr():
    result = score_prompt(["a", "b", "c", "d"], {"c", "z"})
    assert result["hit@1"] == 0.0 and result["hit@3"] == 1.0
    assert result["recall@3"] == 0.5
    assert abs(result["mrr"] - 1 / 3) < 1e-9


def test_evaluate_averages_over_prompts():
    rows = [{"prompt": "postgres schema migration", "needed": ["sql-migrations"]},
            {"prompt": "click buttons on the web page", "needed": ["browser-testing"]}]
    assert evaluate(Bm25(SKILLS), rows)["hit@1"] == 1.0
