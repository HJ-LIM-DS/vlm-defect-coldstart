"""B-교차: 학습 제품의 결함 이미지로 결함 키를 분류하는 로지스틱 회귀. 새 제품 라벨 0장.

fold마다 학습 제품 12개의 test 결함 이미지(키 라벨)로 학습하고, 평가 제품 3개에서는 그 제품 사전의 키 중
분류기가 아는 키(공통 키)만 비교해 고른다. 사전에 분류기가 모르는 키만 있으면 예측 없음(틀린 것으로 채점).
정규화 세기 C는 학습 제품 안에서 제품 단위 교차 검증(GroupKFold)으로 고른다.

사용법:
    python -m src.baselines.cross_product --out reports/predictions/b_cross.jsonl
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from src.baselines.common import FeatureBank, fit_logreg, load_index, predict_restricted, test_defect_rows, write_predictions
from src.data.lexicon import defect_keys, load_lexicon
from src.data.splits import N_FOLDS, load_splits

C_GRID = (0.1, 1.0, 10.0)


def select_C(X, y, groups, seed: int) -> float:
    best, best_acc = C_GRID[0], -1.0
    gkf = GroupKFold(n_splits=min(4, len(set(groups))))
    for C in C_GRID:
        accs = []
        for tr, te in gkf.split(X, y, groups):
            scaler, clf = fit_logreg(X[tr], [y[i] for i in tr], C, seed)
            allowed = sorted(set(y[i] for i in te))
            pred = predict_restricted(scaler, clf, X[te], allowed)
            accs.append(np.mean([p == y[i] for p, i in zip(pred, te)]))
        acc = float(np.mean(accs))
        if acc > best_acc:
            best, best_acc = C, acc
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("reports/predictions/b_cross.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_index()
    lex = load_lexicon()
    keys = defect_keys(lex)
    fold_of = load_splits()
    defect = test_defect_rows(rows)
    bank = FeatureBank(defect)
    all_rows, all_pred, chosen = [], [], {}
    for fold in range(1, N_FOLDS + 1):
        train = [r for r in defect if fold_of[r["category"]] != fold]
        evals = [r for r in defect if fold_of[r["category"]] == fold]
        X = bank.matrix([r["image_id"] for r in train])
        y = [r["defect_type"] for r in train]
        groups = [r["category"] for r in train]
        C = select_C(X, y, groups, args.seed)
        chosen[fold] = C
        scaler, clf = fit_logreg(X, y, C, args.seed)
        for cat in sorted({r["category"] for r in evals}):
            sub = [r for r in evals if r["category"] == cat]
            pred = predict_restricted(scaler, clf, bank.matrix([r["image_id"] for r in sub]), keys[cat])
            all_rows.extend(sub)
            all_pred.extend(pred)
            acc = np.mean([p == r["defect_type"] for p, r in zip(pred, sub)])
            print(f"fold {fold} {cat:<12} C={C:<5} n={len(sub):>4} 유형 정확도 {acc * 100:5.1f}  예측 없음 {sum(p is None for p in pred)}")
    write_predictions(args.out, all_rows, all_pred, {"method": "b_cross"})
    print(f"저장: {args.out} (fold별 C: {chosen})")


if __name__ == "__main__":
    main()
