"""학습 표본 구성의 누수 검사: 평가 제품의 이미지가 학습과 검증에 들어가지 않아야 한다."""

import csv
import json
from pathlib import Path

import pytest

from src.data.lexicon import load_lexicon
from src.data.splits import load_splits
from src.vlm.prompts import parse_output

SFT_DIR = Path("data/sft/f1_nocrop")
INDEX = Path("data/index.csv")
SPLITS = Path("data/splits.csv")

pytestmark = pytest.mark.skipif(not (SFT_DIR / "meta.json").is_file() or not INDEX.is_file(),
                                reason="fold 1 학습 표본이 아직 없다")


def _load(name):
    return [json.loads(x) for x in (SFT_DIR / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def test_no_eval_product_in_train_or_val():
    fold_of = load_splits(SPLITS)
    meta = json.loads((SFT_DIR / "meta.json").read_text(encoding="utf-8"))
    eval_cats = {c for c, f in fold_of.items() if f == meta["fold"]}
    assert set(meta["eval_categories"]) == eval_cats
    for name in ("train", "val"):
        cats = {r["category"] for r in _load(name)}
        assert not (cats & eval_cats), f"{name}에 평가 제품 {cats & eval_cats}"
        for r in _load(name):
            for c in r["messages"][0]["content"]:
                if c["type"] == "image":
                    assert not any(f"/{e}/" in c["image"] for e in eval_cats), f"{name} 이미지 경로에 평가 제품: {c['image']}"


def test_eval_covers_every_test_image_of_fold_products():
    fold_of = load_splits(SPLITS)
    meta = json.loads((SFT_DIR / "meta.json").read_text(encoding="utf-8"))
    eval_cats = {c for c, f in fold_of.items() if f == meta["fold"]}
    with open(INDEX, encoding="utf-8") as f:
        expected = {r["image_id"] for r in csv.DictReader(f) if r["split"] == "test" and r["category"] in eval_cats}
    got = {r["image_id"] for r in _load("eval")}
    assert got == expected


def test_train_val_disjoint_and_targets_parse():
    lex = load_lexicon()
    train, val = _load("train"), _load("val")
    assert not ({r["image_id"] for r in train} & {r["image_id"] for r in val})
    for r in train[:50] + val[:50]:
        p = parse_output(r["target"], lex, r["category"])
        assert p["format_ok"]
        assert p["key"] == (r["defect_type"] or "good")


def test_reference_is_same_product_train_good():
    for r in _load("train")[:200]:
        imgs = [c["image"] for c in r["messages"][0]["content"] if c["type"] == "image"]
        assert imgs[0].endswith(r["image_id"].split("/", 1)[1] + ".png") or r["image_id"].split("/")[0] in imgs[0]
        ref = imgs[-1]
        assert f"/{r['category']}/train/good/" in ref, ref
