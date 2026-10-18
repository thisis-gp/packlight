import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from packlight.registry import Capability, add, content_hash, parse_frontmatter


def test_frontmatter_plain_and_quoted():
    meta, body = parse_frontmatter('---\nname: demo\ndescription: "Use when testing"\n---\nBody text')
    assert meta == {"name": "demo", "description": "Use when testing"}
    assert body == "Body text"


def test_frontmatter_folded_block():
    meta, _ = parse_frontmatter("---\nname: demo\ndescription: >\n  first line\n  second line\n---\nx")
    assert meta["description"] == "first line second line"


def test_no_frontmatter_returns_whole_text():
    meta, body = parse_frontmatter("just a body")
    assert meta == {} and body == "just a body"


def test_same_skill_in_two_places_merges_into_one_record():
    registry: dict[str, Capability] = {}
    add(registry, "skill", "demo", "d", "body", "claude:demo", "archived")
    add(registry, "skill", "demo", "d", "body", "agents:demo", "active")
    assert len(registry) == 1
    only = next(iter(registry.values()))
    assert only.locations == ["claude:demo", "agents:demo"]
    assert only.status == "active"


def test_content_hash_changes_with_body():
    assert content_hash("a", "b", "c") != content_hash("a", "b", "d")
