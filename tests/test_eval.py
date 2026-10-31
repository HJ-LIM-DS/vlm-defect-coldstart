"""지표와 통계 모듈 검사."""

import numpy as np
import pytest

from src.eval.metrics import evaluate, macro_f1
from src.eval.stats import holm, mcnemar, paired_bootstrap, product_mean


def _rows():
    scope = {("a", "crack"): "shared", ("a", "x"): "unique", ("b", "crack"): "shared", ("b", "hole"): "unique"}
    rows = [
        {"image_id": "a1", "category": "a", "label": "defect", "defect_type": "crack", "pred_key": "crack", "format_ok": True},
        {"image_id": "a2", "category": "a", "label": "defect", "defect_type": "crack", "pred_key": "x", "format_ok": True},
        {"image_id": "a3", "category": "a", "label": "defect", "defect_type": "x", "pred_key": "x", "format_ok": True},
        {"image_id": "a4", "category": "a", "label": "defect", "defect_type": "x", "pred_key": None, "format_ok": False},
        {"image_id": "b1", "category": "b", "label": "defect", "defect_type": "crack", "pred_key": "crack", "format_ok": True},
        {"image_id": "g1", "category": "a", "label": "good", "defect_type": "", "pred_key": "good", "format_ok": True},
        {"image_id": "g2", "category": "b", "label": "good", "defect_type": "", "pred_key": "crack", "format_ok": True},
    ]
    return rows, scope


def test_evaluate_basic():
    rows, scope = _rows()
    m = evaluate(rows, scope)
    assert m["n_defect"] == 5 and m["n_good"] == 2
    assert m["type_acc_product_mean"] == pytest.approx((0.5 + 1.0) / 2)
    assert m["type_acc_image_mean"] == pytest.approx(3 / 5)
    assert m["type_acc_shared"] == pytest.approx((0.5 + 1.0) / 2)
    assert m["type_acc_unique"] == pytest.approx(0.5)
    assert m["normal_rate"] == pytest.approx(0.5)
    assert m["format_rate"] == pytest.approx(6 / 7)
    assert m["per_product"] == {"a": 0.5, "b": 1.0}


def test_macro_f1_handles_absent_predictions():
    rows, _ = _rows()
    assert 0.0 <= macro_f1(rows) <= 1.0


def test_single_type_product_excluded_from_type_accuracy():
    rows, scope = _rows()
    scope = dict(scope)
    scope[("t", "only")] = "unique"
    rows = rows + [
        {"image_id": "t1", "category": "t", "label": "defect", "defect_type": "only", "pred_key": "only", "format_ok": True},
        {"image_id": "t2", "category": "t", "label": "good", "defect_type": "", "pred_key": "good", "format_ok": True},
    ]
    m = evaluate(rows, scope)
    assert m["excluded_products"] == ["t"]
    assert m["n_defect"] == 5 and "t" not in m["per_product"]
    assert m["type_acc_product_mean"] == pytest.approx((0.5 + 1.0) / 2)
    assert m["n_good"] == 3 and m["normal_rate"] == pytest.approx(2 / 3)


def test_product_mean_ignores_group_size():
    correct = np.array([1, 1, 1, 1, 0])
    groups = np.array(["a", "a", "a", "a", "b"])
    assert product_mean(correct, groups) == pytest.approx(0.5)


def test_paired_bootstrap_sign_and_width():
    rng = np.random.default_rng(0)
    groups = np.repeat(["a", "b", "c"], 200)
    a = (rng.random(600) < 0.7).astype(float)
    b = (rng.random(600) < 0.5).astype(float)
    r = paired_bootstrap(a, b, groups, n_boot=2000)
    assert r["ci95"][0] < r["diff"] < r["ci95"][1]
    assert r["ci95"][0] > 0
    same = paired_bootstrap(a, a, groups, n_boot=500)
    assert same["diff"] == 0 and same["ci95"] == [0.0, 0.0]


def test_mcnemar_and_holm():
    a = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1], bool)
    b = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], bool)
    r = mcnemar(a, b)
    assert r["n10"] == 8 and r["n01"] == 1 and r["p"] < 0.05
    assert mcnemar(a, a)["p"] == 1.0
    adj = holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert adj["x"] == pytest.approx(0.03) and adj["z"] == pytest.approx(0.06) and adj["y"] == pytest.approx(0.06)
