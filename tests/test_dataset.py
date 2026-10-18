import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from packlight.dataset import clean_prompts, judge_prompt, leaks_name, parse_answer, parse_json_array
from packlight.llm import MissingKeyError, load_key


class FakeGateway:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages, **kwargs):
        return self.reply


def test_parse_json_array_ignores_chatter_and_bad_json():
    assert parse_json_array('Sure!\n["a", "b"]\nHope this helps') == ["a", "b"]
    assert parse_json_array("no array here") == []
    assert parse_json_array("[1, 2,") == []


def test_parse_json_array_survives_sloppy_output():
    assert parse_json_array('["tab	here", "ok"]') == ["tab	here", "ok"]
    assert parse_json_array('["one", "two with \\"quote\\"", "three",]') == ["one", 'two with "quote"', "three"]
    assert parse_json_array("[0, 2, 4,]") == [0, 2, 4]


def test_clean_prompts_dedupes_and_drops_junk():
    assert clean_prompts(["Fix it", "fix it", " ", 5, "ok", "run the tests please"]) == ["Fix it", "run the tests please"]


def test_name_leak_detection():
    assert leaks_name("please use design-standard here", "design-standard")
    assert leaks_name("apply the design standard", "design-standard")
    assert leaks_name("run the humanizer on this", "humanizer")
    assert not leaks_name("make the landing page look less generic", "design-standard")
    assert not leaks_name("humanizing text is hard", "humanizer")


def test_judge_maps_ids_to_names_and_ignores_bad_ids():
    candidates = [{"name": "a", "description": "x"}, {"name": "b", "description": "y"}, {"name": "c", "description": "z"}]
    assert judge_prompt(FakeGateway('reasoning [1] here\nANSWER: [0, 2, 9, "x"]'), "p", candidates) == ["a", "c"]
    assert judge_prompt(FakeGateway("short thought\nANSWER: []"), "p", candidates) == []


def test_judge_without_an_answer_line_is_unjudged_not_empty():
    candidates = [{"name": "a", "description": "x"}]
    assert judge_prompt(FakeGateway("I think [0] but never answered"), "p", candidates) is None
    assert parse_answer("ANSWER: [1]\nlater ANSWER: [2, 3]") == [2, 3]


def test_missing_key_raises(monkeypatch, tmp_path):
    import packlight.llm as llm
    monkeypatch.delenv(llm.KEY_ENV, raising=False)
    monkeypatch.setattr(llm, "KEY_FILE", tmp_path / "nokey")
    try:
        load_key()
    except MissingKeyError:
        return
    raise AssertionError("expected MissingKeyError")
