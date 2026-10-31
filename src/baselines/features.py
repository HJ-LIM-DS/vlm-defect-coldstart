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


class PixelDataset(torch.utils.data.Dataset):
    """이미지 디코드와 프로세서 전처리를 DataLoader 워커에서 돌린다(디코드가 병목)."""

    def __init__(self, paths: list[Path], proc):
        self.paths, self.proc = paths, proc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.proc(images=img, return_tensors="pt")["pixel_values"][0]


@torch.no_grad()
def extract(model_name: str, rows: list[dict], root: Path, batch_size: int = 32, device: str = "cuda",
            source: str = "image", crop_root: Path = Path("data/patchcore"), num_workers: int = 4) -> np.ndarray:
    from transformers import AutoImageProcessor, AutoModel

    hf_id = MODELS[model_name]
    proc = AutoImageProcessor.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id).to(device).eval()
    paths = [image_path(r, root, source, crop_root) for r in rows]
    loader = torch.utils.data.DataLoader(PixelDataset(paths, proc), batch_size=batch_size, num_workers=num_workers)
    feats, done = [], 0
    for pixel_values in loader:
        pixel_values = pixel_values.to(device)
        if model_name == "dinov2":
            h = model(pixel_values=pixel_values).last_hidden_state
            f = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
        else:
            f = model.get_image_features(pixel_values=pixel_values)
        f = torch.nn.functional.normalize(f.float(), dim=1)
        feats.append(f.cpu().numpy())
        done += len(pixel_values)
        print(f"\r{model_name} {done}/{len(rows)}", end="", flush=True)
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
