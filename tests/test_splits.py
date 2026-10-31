"""제품 5-fold 분할 규칙 검사."""

from collections import Counter
from pathlib import Path

from src.data.lexicon import key_products, load_lexicon
from src.data.splits import N_FOLDS, load_splits, make_splits, shared_keys_survive


def test_rules_hold():
    lex = load_lexicon()
    fold_of = make_splits(lex)
    assert len(fold_of) == 15
    for f in range(1, N_FOLDS + 1):
        members = [c for c, v in fold_of.items() if v == f]
        kinds = Counter(lex["products"][c]["kind"] for c in members)
        assert kinds == Counter({"object": 2, "texture": 1}), f"fold {f}: {members}"
    assert shared_keys_survive({c: v - 1 for c, v in fold_of.items()}, key_products(lex))


def test_deterministic_and_matches_file():
    lex = load_lexicon()
    assert make_splits(lex) == make_splits(lex)
    path = Path("data/splits.csv")
    if path.is_file():
        assert load_splits(path) == make_splits(lex)
