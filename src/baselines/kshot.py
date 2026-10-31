"""B-k장과 B-전체: 새 제품의 결함 라벨을 유형당 k장(1, 2, 5, 10) 또는 전부 쓰는 로지스틱 회귀.

B-k장: 제품마다 결함 유형당 k장을 뽑아 학습하고 뽑지 않은 결함 이미지를 예측한다. 뽑기를 seed 0~19로 20회 반복한다.
  유형의 장수가 k 이하이면 (장수 - 1)장만 뽑아 적어도 한 장은 평가에 남긴다.
  VLM과 비교할 때는 반복마다 뽑지 않은 이미지 집합으로 VLM도 다시 채점한다.
B-전체: 제품 안에서 유형 층화 5-fold 교차 예측으로 모든 결함 이미지에 예측을 단다.

출력: reports/predictions/b_kshot_k{k}_rep{r}.jsonl, reports/predictions/b_full.jsonl

사용법:
    python -m src.baselines.kshot --ks 1 2 5 10 --reps 20
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.baselines.common import FeatureBank, fit_logreg, load_index, predict_restricted, test_defect_rows, write_predictions
from src.data.lexicon import defect_keys, load_lexicon

C_FIXED = 1.0  # 라벨이 적어 교차 검증으로 고를 수 없으므로 고정한다


def kshot_once(sub: list[dict], bank: FeatureBank, allowed: list[str], k: int, seed: int):
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for r in sub:
        by_type[r["defect_type"]].append(r)
    train, held = [], []
    for t, items in by_type.items():
        items = sorted(items, key=lambda r: r["image_id"])
        rng.shuffle(items)
        n = min(k, len(items) - 1)
        train.extend(items[:n])
        held.extend(items[n:])
    if len({r["defect_type"] for r in train}) < 2:
        # 유형이 하나뿐인 제품은 분류할 것이 없다. 그 유형을 그대로 낸다.
        return held, [train[0]["defect_type"]] * len(held)
    scaler, clf = fit_logreg(bank.matrix([r["image_id"] for r in train]), [r["defect_type"] for r in train], C_FIXED, seed)
    pred = predict_restricted(scaler, clf, bank.matrix([r["image_id"] for r in held]), allowed)
    return held, pred


def full_cv(sub: list[dict], bank: FeatureBank, allowed: list[str], seed: int = 0, n_splits: int = 5):
    y = [r["defect_type"] for r in sub]
    if len(set(y)) < 2:
        return sub, list(y)
    X = bank.matrix([r["image_id"] for r in sub])
    counts = defaultdict(int)
    for t in y:
        counts[t] += 1
    n_splits = max(2, min(n_splits, min(counts.values())))
    pred = [None] * len(sub)
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y):
        scaler, clf = fit_logreg(X[tr], [y[i] for i in tr], C_FIXED, seed)
        for i, p in zip(te, predict_restricted(scaler, clf, X[te], allowed)):
            pred[i] = p
    return sub, pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 10])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/predictions"))
    args = ap.parse_args()

    rows = load_index()
    lex = load_lexicon()
    keys = defect_keys(lex)
    defect = test_defect_rows(rows)
    bank = FeatureBank(defect)
    cats = sorted({r["category"] for r in defect})

    for k in args.ks:
        accs = []
        for rep in range(args.reps):
            out_rows, out_pred = [], []
            for cat in cats:
                sub = [r for r in defect if r["category"] == cat]
                held, pred = kshot_once(sub, bank, keys[cat], k, seed=rep)
                out_rows.extend(held)
                out_pred.extend(pred)
            write_predictions(args.out_dir / f"b_kshot_k{k}_rep{rep}.jsonl", out_rows, out_pred, {"method": f"b_kshot_k{k}", "rep": rep})
            accs.append(np.mean([p == r["defect_type"] for p, r in zip(out_pred, out_rows)]))
        print(f"k={k:<3} 이미지 평균 정확도 {np.mean(accs) * 100:5.1f} (반복 {args.reps}회 표준편차 {np.std(accs) * 100:4.1f})")

    out_rows, out_pred = [], []
    for cat in cats:
        sub = [r for r in defect if r["category"] == cat]
        held, pred = full_cv(sub, bank, keys[cat])
        out_rows.extend(held)
        out_pred.extend(pred)
    write_predictions(args.out_dir / "b_full.jsonl", out_rows, out_pred, {"method": "b_full"})
    print(f"전체 라벨 교차 예측 이미지 평균 정확도 {np.mean([p == r['defect_type'] for p, r in zip(out_pred, out_rows)]) * 100:5.1f}")


if __name__ == "__main__":
    main()
