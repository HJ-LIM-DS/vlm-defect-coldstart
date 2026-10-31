"""B-CLIP: SigLIP 텍스트 매칭. 새 제품 라벨 0장, 학습 없음.

결함 키마다 영어 문장 "a photo of a {제품} with {정의}" 를 텍스트 임베딩으로 만들고, 이미지 임베딩과의 코사인 유사도가
가장 높은 키를 고른다. 이미지 임베딩은 전체 이미지(siglip.npz)와 이상 영역 crop(siglip_crop.npz) 두 변형을 모두 계산하고
둘 다 저장한다. 어느 변형을 대표로 쓸지는 학습 제품(평가 fold 밖)의 정확도로 fold마다 고른다.

사용법:
    python -m src.baselines.clip_match --out reports/predictions/b_clip.jsonl
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.baselines.common import load_index, test_defect_rows, write_predictions
from src.baselines.features import MODELS, load_features
from src.data.lexicon import load_lexicon
from src.data.splits import N_FOLDS, load_splits

TEMPLATE = "a photo of a {product} with {definition}"


@torch.no_grad()
def text_embeddings(lex: dict, device: str = "cuda") -> dict[tuple[str, str], np.ndarray]:
    from transformers import AutoModel, AutoTokenizer

    hf_id = MODELS["siglip"]
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id).to(device).eval()
    pairs, texts = [], []
    for cat, p in lex["products"].items():
        for key, d in p["defects"].items():
            pairs.append((cat, key))
            texts.append(TEMPLATE.format(product=p["name_en"], definition=d["definition_en"]))
    inputs = tok(texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(device)
    emb = model.get_text_features(**inputs)
    emb = torch.nn.functional.normalize(emb.float(), dim=1).cpu().numpy()
    return {p: e for p, e in zip(pairs, emb)}


def match(rows: list[dict], image_feats: dict[str, np.ndarray], text: dict, lex: dict) -> list[str]:
    preds = []
    for r in rows:
        cat = r["category"]
        keys = list(lex["products"][cat]["defects"])
        T = np.stack([text[(cat, k)] for k in keys])
        preds.append(keys[int(np.argmax(T @ image_feats[r["image_id"]]))])
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("reports/predictions/b_clip.jsonl"))
    args = ap.parse_args()

    rows = load_index()
    lex = load_lexicon()
    fold_of = load_splits()
    defect = test_defect_rows(rows)
    text = text_embeddings(lex)
    variants = {}
    for name in ("siglip", "siglip_crop"):
        ids, feat = load_features(Path("data/features") / f"{name}.npz")
        variants[name] = dict(zip(ids, feat))
    preds = {name: match(defect, variants[name], text, lex) for name in variants}
    correct = {name: np.array([p == r["defect_type"] for p, r in zip(preds[name], defect)]) for name in variants}
    cats = np.array([r["category"] for r in defect])

    final = [None] * len(defect)
    for fold in range(1, N_FOLDS + 1):
        train_mask = np.array([fold_of[c] != fold for c in cats])
        best = max(variants, key=lambda n: correct[n][train_mask].mean())
        for i in np.where(~train_mask)[0]:
            final[i] = preds[best][i]
        for cat in sorted({c for c in cats if fold_of[c] == fold}):
            m = cats == cat
            print(f"fold {fold} {cat:<12} 대표={best:<12} 유형 정확도 전체 {correct['siglip'][m].mean() * 100:5.1f} "
                  f"crop {correct['siglip_crop'][m].mean() * 100:5.1f}")
    write_predictions(args.out, defect, final, {"method": "b_clip"})
    for name in variants:
        write_predictions(args.out.with_name(f"b_clip_{name}.jsonl"), defect, preds[name], {"method": f"b_clip_{name}"})
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
