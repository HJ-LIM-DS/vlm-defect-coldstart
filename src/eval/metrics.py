"""예측 기록을 지표로 바꾼다.

예측 기록의 한 행: image_id, category, label(good|defect), defect_type(정답 키, 정상이면 빈 문자열),
pred_key(예측 키, 정상이면 'good', 형식 위반이면 None), format_ok(bool).

지표
- type_accuracy: 결함 이미지에서 pred_key == defect_type 비율. 제품별로 구해 평균(주 지표)하고 이미지 평균도 함께 둔다.
- shared/unique: 공통 유형과 고유 유형으로 나눈 정확도(제품 평균).
- macro_f1: 유형(제품, 키)별 F1의 평균.
- normal_rate: 정상 이미지에 'good'이라 답한 비율.
- format_rate: 전체 이미지의 형식 준수율.
"""

from collections import defaultdict

import numpy as np

from src.eval.stats import product_mean


def single_type_products(scope: dict[tuple[str, str], str]) -> set[str]:
    """결함 유형이 하나뿐인 제품(유형 분류가 자명하므로 유형 정확도 평균에서 뺀다)."""
    n = defaultdict(int)
    for cat, _ in scope:
        n[cat] += 1
    return {cat for cat, k in n.items() if k == 1}


def correctness(rows: list[dict], scope: dict[tuple[str, str], str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """결함 이미지만 골라 (정오 벡터, 제품 벡터)를 돌려준다. 다른 방법과 짝지어 비교할 때 쓴다.

    scope를 주면 유형이 하나뿐인 제품을 제외한다.
    """
    skip = single_type_products(scope) if scope else set()
    defect = [r for r in rows if r["label"] == "defect" and r["category"] not in skip]
    correct = np.array([r["pred_key"] == r["defect_type"] for r in defect], dtype=float)
    groups = np.array([r["category"] for r in defect])
    return correct, groups


def macro_f1(rows: list[dict]) -> float:
    defect = [r for r in rows if r["label"] == "defect"]
    types = sorted({(r["category"], r["defect_type"]) for r in defect})
    f1s = []
    for cat, key in types:
        tp = sum(r["category"] == cat and r["defect_type"] == key and r["pred_key"] == key for r in defect)
        fp = sum(r["category"] == cat and r["defect_type"] != key and r["pred_key"] == key for r in defect)
        fn = sum(r["category"] == cat and r["defect_type"] == key and r["pred_key"] != key for r in defect)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def evaluate(rows: list[dict], scope: dict[tuple[str, str], str]) -> dict:
    """scope는 src.data.lexicon.type_scope 의 결과. 유형 정확도는 유형이 하나뿐인 제품을 뺀 제품 평균이다."""
    correct, groups = correctness(rows, scope)
    skip = single_type_products(scope)
    defect = [r for r in rows if r["label"] == "defect" and r["category"] not in skip]
    out = {
        "n_defect": int(len(defect)),
        "excluded_products": sorted(skip),
        "type_acc_product_mean": product_mean(correct, groups) if len(defect) else float("nan"),
        "type_acc_image_mean": float(correct.mean()) if len(defect) else float("nan"),
        "macro_f1": macro_f1(rows),
    }
    for name in ("shared", "unique"):
        mask = np.array([scope[(r["category"], r["defect_type"])] == name for r in defect], dtype=bool)
        out[f"type_acc_{name}"] = product_mean(correct[mask], groups[mask]) if mask.any() else float("nan")
        out[f"n_{name}"] = int(mask.sum())
    good = [r for r in rows if r["label"] == "good"]
    out["n_good"] = int(len(good))
    out["normal_rate"] = float(np.mean([r["pred_key"] == "good" for r in good])) if good else float("nan")
    out["format_rate"] = float(np.mean([bool(r["format_ok"]) for r in rows])) if rows else float("nan")
    per_product = defaultdict(list)
    for r, c in zip(defect, correct):
        per_product[r["category"]].append(c)
    out["per_product"] = {k: float(np.mean(v)) for k, v in sorted(per_product.items())}
    return out
