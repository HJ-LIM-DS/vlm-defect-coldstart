"""이미지 특징을 추출해 캐시한다(data/features/<model>[_crop].npz).

비-VLM 기준선(교차 제품 분류기, few-shot 분류기, 전체 라벨 분류기)과 정상 참조 선택이 같은 특징을 쓴다.
- dinov2: facebook/dinov2-base 의 CLS 토큰과 패치 토큰 평균을 이어 붙인 1536차원
- siglip: google/siglip-base-patch16-224 이미지 임베딩(텍스트 매칭용, 768차원)
- --source crop: 전체 이미지 대신 PatchCore 이상 영역 crop(data/patchcore/<category>/crops/)을 넣는다. test 이미지만 있다.

사용법:
    python -m src.baselines.features --model dinov2
    python -m src.baselines.features --model dinov2 --source crop
    python -m src.baselines.features --model siglip
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODELS = {
    "dinov2": "facebook/dinov2-base",
    "siglip": "google/siglip-base-patch16-224",
}


def load_index(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def image_path(r: dict, root: Path, source: str, crop_root: Path) -> Path:
    if source == "crop":
        return crop_root / r["category"] / "crops" / (r["image_id"].replace("/", "__") + ".png")
    return root / r["rel_path"]


@torch.no_grad()
def extract(model_name: str, rows: list[dict], root: Path, batch_size: int = 32, device: str = "cuda",
            source: str = "image", crop_root: Path = Path("data/patchcore")) -> np.ndarray:
    from transformers import AutoImageProcessor, AutoModel

    hf_id = MODELS[model_name]
    proc = AutoImageProcessor.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id).to(device).eval()
    feats = []
    for i in range(0, len(rows), batch_size):
        imgs = [Image.open(image_path(r, root, source, crop_root)).convert("RGB") for r in rows[i:i + batch_size]]
        inputs = proc(images=imgs, return_tensors="pt").to(device)
        if model_name == "dinov2":
            h = model(**inputs).last_hidden_state
            f = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
        else:
            f = model.get_image_features(**inputs)
        f = torch.nn.functional.normalize(f.float(), dim=1)
        feats.append(f.cpu().numpy())
        print(f"\r{model_name} {min(i + batch_size, len(rows))}/{len(rows)}", end="", flush=True)
    print()
    return np.concatenate(feats)


def load_features(path: Path) -> tuple[list[str], np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return list(z["image_id"]), z["feat"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--index", type=Path, default=Path("data/index.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/mvtec_ad"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--source", choices=["image", "crop"], default="image")
    ap.add_argument("--crop-root", type=Path, default=Path("data/patchcore"))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    suffix = "_crop" if args.source == "crop" else ""
    out = args.out or Path(f"data/features/{args.model}{suffix}.npz")

    rows = load_index(args.index)
    if args.source == "crop":
        rows = [r for r in rows if r["split"] == "test"]
    feat = extract(args.model, rows, args.root, args.batch_size, source=args.source, crop_root=args.crop_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, image_id=np.array([r["image_id"] for r in rows]), feat=feat.astype(np.float32))
    print(f"{feat.shape} 저장: {out}")


if __name__ == "__main__":
    main()
