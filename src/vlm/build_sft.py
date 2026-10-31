"""fold별 LoRA 학습 표본(JSONL)을 만든다.

fold f를 평가할 때의 학습 제품은 나머지 12개다. 표본은 학습 제품의
- test 결함 이미지 전부(정답: 유형 이름과 근거. 근거는 사전의 정의 문장을 그대로 쓴다)
- test 정상 이미지 전부(정답: '유형: 정상')
이고, 학습 제품 안에서 제품 층화로 15%를 검증용(inner-val)으로 뗀다. 학습 설정은 이 검증 분할로만 고른다.
평가 제품의 이미지는 어떤 형태로도 들어가지 않는다.

입력 구성 옵션(입력 제거 실험용): --no-crop, --no-ref, --no-definitions.
crop은 data/patchcore/<category>/crops/, 정상 참조는 DINOv2 특징이 가장 가까운 같은 제품 train/good 이미지다.

사용법:
    python -m src.vlm.build_sft --fold 1 --out data/sft/f1
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from src.baselines.features import load_features
from src.data.lexicon import load_lexicon
from src.data.splits import load_splits
from src.vlm.prompts import build_messages, target_text

VAL_FRAC = 0.15


def nearest_good_reference(feat_ids: list[str], feats: np.ndarray, rows: list[dict], root: Path) -> dict[str, str]:
    """이미지 ID -> 같은 제품 train/good 중 코사인 유사도가 가장 높은 이미지의 경로(프로젝트 루트 기준)."""
    pos = {i: k for k, i in enumerate(feat_ids)}
    by_cat = {}
    for r in rows:
        if r["split"] == "train":
            by_cat.setdefault(r["category"], []).append(r)
    out = {}
    for cat, goods in by_cat.items():
        gid = [pos[r["image_id"]] for r in goods]
        gfeat = feats[gid]
        for r in rows:
            if r["category"] != cat:
                continue
            sims = gfeat @ feats[pos[r["image_id"]]]
            if r["split"] == "train":
                sims[gid.index(pos[r["image_id"]])] = -1  # 자기 자신은 제외
            out[r["image_id"]] = (root / goods[int(np.argmax(sims))]["rel_path"]).as_posix()
    return out


def crop_path(crop_root: Path, image_id: str) -> str:
    return (crop_root / image_id.split("/")[0] / "crops" / (image_id.replace("/", "__") + ".png")).as_posix()


def make_sample(r: dict, lex: dict, root: Path, refs: dict[str, str], crop_root: Path,
                use_crop: bool, use_ref: bool, with_definitions: bool, seed: int) -> dict:
    key = r["defect_type"] or None
    return {
        "image_id": r["image_id"],
        "category": r["category"],
        "label": r["label"],
        "defect_type": r["defect_type"],
        "messages": build_messages(
            lex, r["category"], (root / r["rel_path"]).as_posix(),
            crop=crop_path(crop_root, r["image_id"]) if use_crop else None,
            ref=refs[r["image_id"]] if use_ref else None,
            with_definitions=with_definitions, seed=seed),
        "target": target_text(lex, r["category"], key),
    }


def build(fold: int, rows: list[dict], lex: dict, fold_of: dict[str, int], refs: dict[str, str], root: Path,
          crop_root: Path, use_crop: bool, use_ref: bool, with_definitions: bool, seed: int
          ) -> tuple[list[dict], list[dict], list[dict]]:
    """(학습, 검증, 평가) 표본. 평가 표본은 fold에 든 제품의 test 이미지 전부다."""
    rng = random.Random(seed)
    train_cats = sorted(c for c, f in fold_of.items() if f != fold)
    eval_cats = sorted(c for c, f in fold_of.items() if f == fold)
    samples, evals = [], []
    for r in rows:
        if r["split"] != "test":
            continue
        s = make_sample(r, lex, root, refs, crop_root, use_crop, use_ref, with_definitions, rng.randrange(1 << 30))
        if r["category"] in train_cats:
            samples.append(s)
        elif r["category"] in eval_cats:
            evals.append(s)
    train, val = [], []
    for cat in train_cats:
        sub = [s for s in samples if s["category"] == cat]
        rng.shuffle(sub)
        n_val = max(1, int(round(len(sub) * VAL_FRAC)))
        val.extend(sub[:n_val])
        train.extend(sub[n_val:])
    rng.shuffle(train)
    return train, val, evals


def write_jsonl(path: Path, items: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--index", type=Path, default=Path("data/index.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/mvtec_ad"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    ap.add_argument("--features", type=Path, default=Path("data/features/dinov2.npz"))
    ap.add_argument("--crop-root", type=Path, default=Path("data/patchcore"))
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--no-definitions", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.index, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lex = load_lexicon()
    fold_of = load_splits(args.splits)
    ids, feats = load_features(args.features)
    refs = nearest_good_reference(ids, feats, rows, args.root)
    train, val, evals = build(args.fold, rows, lex, fold_of, refs, args.root, args.crop_root,
                              not args.no_crop, not args.no_ref, not args.no_definitions, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "val.jsonl", val)
    write_jsonl(args.out / "eval.jsonl", evals)
    meta = {"fold": args.fold, "crop": not args.no_crop, "ref": not args.no_ref,
            "definitions": not args.no_definitions, "seed": args.seed,
            "n_train": len(train), "n_val": len(val), "n_eval": len(evals),
            "train_categories": sorted(c for c, f in fold_of.items() if f != args.fold),
            "eval_categories": sorted(c for c, f in fold_of.items() if f == args.fold)}
    with open(args.out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    n_def = sum(s["label"] == "defect" for s in train)
    print(f"fold {args.fold}: 학습 {len(train)}장(결함 {n_def}, 정상 {len(train) - n_def}), "
          f"검증 {len(val)}장, 평가 {len(evals)}장, 저장 {args.out}")


if __name__ == "__main__":
    main()
