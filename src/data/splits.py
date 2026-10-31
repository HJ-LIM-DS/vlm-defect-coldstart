"""제품 단위 5-fold 분할(data/splits.csv)을 만든다.

평가 단위는 이미지가 아니라 제품이다. fold에 든 제품은 학습에서 완전히 빠지고(이미지, 결함 사전 항목 모두),
그 fold를 평가할 때만 결함 사전과 정상 이미지가 주어진다.

배치 규칙(결과를 보기 전에 고정):
1. 텍스처 5개(carpet, grid, leather, tile, wood)는 알파벳 순으로 fold 1부터 5에 하나씩 둔다.
2. 객체 10개는 fold마다 2개씩 둔다.
3. 공통 유형(두 제품 이상이 가진 키)은 어느 fold를 평가하든 학습 측에 그 키를 가진 제품이 하나 이상 남아야 한다.
   그래야 "다른 제품에서 배운 어휘의 전이"를 모든 공통 유형에서 잴 수 있다.
4. 규칙 1~3을 만족하는 배치가 여럿이면 객체를 알파벳 순으로 늘어놓았을 때의 fold 번호 열이 사전순으로 가장 앞선 것을 고른다.
   사람이 고르지 않는다는 뜻이다.

사용법:
    python -m src.data.splits --out data/splits.csv
"""

import argparse
import csv
from pathlib import Path

from src.data.lexicon import DEFAULT_PATH, key_products, load_lexicon

N_FOLDS = 5


def _assignments(n_items: int, per_fold: int):
    """n_items개를 fold마다 per_fold개씩 배정하는 모든 방법을 사전순으로 생성한다."""
    counts = [0] * N_FOLDS
    cur = []

    def rec(i):
        if i == n_items:
            yield tuple(cur)
            return
        for f in range(N_FOLDS):
            if counts[f] < per_fold:
                counts[f] += 1
                cur.append(f)
                yield from rec(i + 1)
                cur.pop()
                counts[f] -= 1

    yield from rec(0)


def shared_keys_survive(fold_of: dict[str, int], kp: dict[str, list[str]]) -> bool:
    for key, cats in kp.items():
        if len(cats) < 2:
            continue
        folds = {fold_of[c] for c in cats}
        if len(folds) == 1:  # 그 키를 가진 제품이 모두 한 fold에 몰리면, 그 fold 평가 때 학습 측에 키가 없다
            return False
    return True


def make_splits(lex: dict) -> dict[str, int]:
    products = lex["products"]
    textures = sorted(c for c, p in products.items() if p["kind"] == "texture")
    objects = sorted(c for c, p in products.items() if p["kind"] == "object")
    if len(textures) != N_FOLDS or len(objects) != 2 * N_FOLDS:
        raise ValueError(f"텍스처 {len(textures)}개, 객체 {len(objects)}개는 이 규칙에 맞지 않는다")
    kp = key_products(lex)
    base = {c: i for i, c in enumerate(textures)}
    for assign in _assignments(len(objects), 2):
        fold_of = dict(base)
        fold_of.update({c: f for c, f in zip(objects, assign)})
        if shared_keys_survive(fold_of, kp):
            return {c: f + 1 for c, f in fold_of.items()}
    raise RuntimeError("규칙을 만족하는 배치가 없다")


def load_splits(path: Path = Path("data/splits.csv")) -> dict[str, int]:
    with open(path, encoding="utf-8") as f:
        return {r["category"]: int(r["fold"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--out", type=Path, default=Path("data/splits.csv"))
    args = ap.parse_args()

    lex = load_lexicon(args.lexicon)
    fold_of = make_splits(lex)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["category", "kind", "fold"])
        for c in sorted(fold_of):
            w.writerow([c, lex["products"][c]["kind"], fold_of[c]])
    for f in range(1, N_FOLDS + 1):
        print(f"fold {f}: " + ", ".join(sorted(c for c, v in fold_of.items() if v == f)))
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
