import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from packlight.split import assign_splits, dedupe_by_name, group_skills


def skill(name, description, body=10):
    return {"name": name, "description": description, "tokens_body": body, "tokens_description": 5}


def test_same_name_keeps_richest_version():
    kept = dedupe_by_name([skill("a", "x", body=5), skill("A", "y", body=50)])
    assert len(kept) == 1 and kept[0]["description"] == "y"


def test_similar_descriptions_share_a_group_and_distinct_ones_do_not():
    skills = [
        skill("video-captions", "Add captions and subtitles to a talking head video"),
        skill("video-subtitles", "Add subtitles and captions to talking head videos"),
        skill("sql-migrations", "Write reversible database schema migrations for postgres"),
    ]
    groups = group_skills(skills)
    assert sorted(map(sorted, groups)) == [[0, 1], [2]]


def test_whole_groups_stay_in_one_split_and_seed_is_deterministic():
    groups = [[i, i + 1] for i in range(0, 40, 2)]
    first = assign_splits(groups, 40, seed=1)
    assert first == assign_splits(groups, 40, seed=1)
    for group in groups:
        assert len({first[i] for i in group}) == 1
    assert {"train", "dev", "test"} <= set(first)


def test_small_prefix_family_is_grouped_even_when_descriptions_differ():
    skills = [
        skill("hyper-audio", "Mix music and voiceover levels"),
        skill("hyper-keyframes", "Zoom and pan camera moves"),
        skill("sql-migrations", "Write reversible database schema migrations for postgres"),
    ]
    groups = group_skills(skills)
    assert sorted(map(sorted, groups)) == [[0, 1], [2]]
