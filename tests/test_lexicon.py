"""결함 사전과 프롬프트 파서의 일관성 검사."""

import random

import pytest

from src.data.lexicon import defect_keys, key_products, load_lexicon, name_to_key, type_scope
from src.vlm.prompts import NORMAL, build_messages, parse_output, target_text


@pytest.fixture(scope="module")
def lex():
    return load_lexicon()


def test_counts(lex):
    assert len(lex["products"]) == 15
    scope = type_scope(lex)
    assert len(scope) == 73
    assert sum(v == "shared" for v in scope.values()) == 39
    assert sum(len(c) >= 2 for c in key_products(lex).values()) == 14


def test_names_unique_within_product(lex):
    for cat in lex["products"]:
        table = name_to_key(lex, cat)
        assert len(table) == len(defect_keys(lex)[cat])
        assert NORMAL not in table


def test_definitions_are_single_sentences(lex):
    for cat, p in lex["products"].items():
        for key, d in p["defects"].items():
            assert d["definition"].endswith("다."), f"{cat}/{key}"
            assert len(d["definition"]) <= 60, f"{cat}/{key} 정의가 너무 길다"
            assert d["definition"].count("다.") == 1, f"{cat}/{key} 두 문장 이상"


def test_target_round_trip(lex):
    for cat, p in lex["products"].items():
        for key in p["defects"]:
            parsed = parse_output(target_text(lex, cat, key), lex, cat)
            assert parsed["key"] == key and parsed["format_ok"], f"{cat}/{key}"
        parsed = parse_output(target_text(lex, cat, None), lex, cat)
        assert parsed["key"] == "good" and parsed["format_ok"]


def test_parse_rejects_unknown_name(lex):
    parsed = parse_output("유형: 없는 유형\n근거: 아무 말", lex, "bottle")
    assert parsed["key"] is None and not parsed["format_ok"]
    parsed = parse_output("결함이 있는 것 같다", lex, "bottle")
    assert not parsed["format_ok"]


def test_prompt_lists_every_type_in_random_order(lex):
    for cat, p in lex["products"].items():
        msgs = build_messages(lex, cat, "img.png", crop="crop.png", ref="ref.png", seed=1)
        text = msgs[0]["content"][-1]["text"]
        for d in p["defects"].values():
            assert f"- {d['name_ko']}: {d['definition']}" in text
        assert sum(c["type"] == "image" for c in msgs[0]["content"]) == 3
    a = build_messages(lex, "pill", "img.png", seed=1)[0]["content"][-1]["text"]
    b = build_messages(lex, "pill", "img.png", seed=2)[0]["content"][-1]["text"]
    assert a != b
    no_def = build_messages(lex, "pill", "img.png", with_definitions=False)[0]["content"][-1]["text"]
    assert "균열" in no_def and "금이 가거나" not in no_def
