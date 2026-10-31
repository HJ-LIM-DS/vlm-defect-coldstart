"""PatchCore(Roth et al., CVPR 2022) 이상 맵과 이상 영역 crop.

라벨 없이 평가 제품의 train/good 만으로 메모리 뱅크를 만든다. 콜드 스타트에서 새 제품의 정상 이미지는
쉽게 모을 수 있다고 가정하므로 이 입력은 모든 비교 대상에 허용한다.

- 백본: torchvision wide_resnet50_2(ImageNet 가중치). layer2와 layer3 특징을 layer2 해상도로 맞춰 이어 붙이고
  3x3 평균으로 이웃 패치를 모은다. 입력은 320x320으로 축소하고 잘라 내지 않는다(결함이 가장자리에 있을 수 있다).
- 메모리 뱅크: 정상 패치 특징을 128차원으로 무작위 투영한 뒤 greedy coreset 으로 1%까지 줄인다.
- 점수: 패치마다 뱅크 최근접 거리(faiss, L2). 이미지 점수는 최댓값. 맵은 원본 해상도로 보간한 뒤 가우시안 블러.
- crop: 맵 최댓값 위치를 중심으로 긴 변의 1/3 길이 정사각형을 잘라 낸다(경계 안으로 밀어 넣음).

산출물(제품별): data/patchcore/<category>/scores.csv(image_id, label, score),
maps/<id>.npy(float16, 원본 해상도), crops/<id>.png

사용법:
    python -m src.detector.patchcore --category bottle
    python -m src.detector.patchcore --all
"""

import argparse
import csv
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from src.data.index_mvtec import CATEGORIES

INPUT_SIZE = 320
PROJ_DIM = 128
CORESET_RATIO = 0.01
BLUR_SIGMA = 4.0
CROP_FRAC = 1 / 3


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        self.pool = torch.nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        x = self.stem(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([self.pool(f2), self.pool(f3)], dim=1)  # (B, 1536, 40, 40)
        return f


TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ImageFolderDataset(torch.utils.data.Dataset):
    """경로 목록을 TRANSFORM 텐서로. 이미지 디코드가 병목이라 DataLoader 워커에서 돌린다."""

    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return TRANSFORM(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def patch_features(model: Backbone, paths: list[Path], device: str, batch_size: int = 16, num_workers: int = 4):
    """(N, H*W, C) float32 와 (H, W)."""
    loader = torch.utils.data.DataLoader(ImageFolderDataset(paths), batch_size=batch_size, num_workers=num_workers,
                                         persistent_workers=False)
    out, hw = [], None
    for x in loader:
        f = model(x.to(device))
        hw = f.shape[-2:]
        out.append(f.permute(0, 2, 3, 1).reshape(f.shape[0], -1, f.shape[1]).cpu())
    return torch.cat(out).numpy(), tuple(hw)


def greedy_coreset(feats: np.ndarray, ratio: float, device: str, seed: int = 0) -> np.ndarray:
    """투영 공간에서 가장 먼 점을 차례로 고르는 greedy k-center. 고른 인덱스를 돌려준다."""
    n = len(feats)
    k = max(1, int(n * ratio))
    g = torch.Generator(device="cpu").manual_seed(seed)
    proj = torch.randn(feats.shape[1], PROJ_DIM, generator=g)
    x = torch.from_numpy(feats).to(device) @ proj.to(device)
    chosen = [int(torch.randint(n, (1,), generator=g))]
    min_d = torch.cdist(x, x[chosen[-1]][None]).squeeze(1)
    for _ in range(k - 1):
        nxt = int(torch.argmax(min_d))
        chosen.append(nxt)
        d = torch.cdist(x, x[nxt][None]).squeeze(1)
        min_d = torch.minimum(min_d, d)
    return np.array(chosen)


def build_bank(model: Backbone, good_paths: list[Path], device: str) -> faiss.Index:
    feats, _ = patch_features(model, good_paths, device)
    feats = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
    idx = greedy_coreset(feats, CORESET_RATIO, device)
    bank = np.ascontiguousarray(feats[idx])
    index = faiss.IndexFlatL2(bank.shape[1])
    index.add(bank)
    return index


def anomaly_maps(model: Backbone, index: faiss.Index, paths: list[Path], device: str):
    """각 이미지의 (원본 해상도 맵, 이미지 점수)."""
    feats, (h, w) = patch_features(model, paths, device)
    results = []
    for p, f in zip(paths, feats):
        d, _ = index.search(np.ascontiguousarray(f.astype(np.float32)), 1)
        m = np.sqrt(d[:, 0]).reshape(h, w)
        with Image.open(p) as im:
            W, H = im.size
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        m = cv2.GaussianBlur(m, (0, 0), BLUR_SIGMA)
        results.append((m, float(m.max())))
    return results


def crop_at_peak(image_path: Path, amap: np.ndarray) -> Image.Image:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        W, H = im.size
    side = int(round(max(W, H) * CROP_FRAC))
    y, x = np.unravel_index(int(np.argmax(amap)), amap.shape)
    x0 = min(max(0, x - side // 2), W - side)
    y0 = min(max(0, y - side // 2), H - side)
    return im.crop((x0, y0, x0 + side, y0 + side))


def run_category(category: str, root: Path, out_root: Path, rows: list[dict], device: str, model: Backbone):
    sub = [r for r in rows if r["category"] == category]
    good = [root / r["rel_path"] for r in sub if r["split"] == "train"]
    test = [r for r in sub if r["split"] == "test"]
    out = out_root / category
    (out / "maps").mkdir(parents=True, exist_ok=True)
    (out / "crops").mkdir(parents=True, exist_ok=True)

    index = build_bank(model, good, device)
    maps = anomaly_maps(model, index, [root / r["rel_path"] for r in test], device)
    with open(out / "scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["image_id", "label", "defect_type", "score"])
        for r, (m, s) in zip(test, maps):
            stem = r["image_id"].replace("/", "__")
            np.save(out / "maps" / f"{stem}.npy", m.astype(np.float16))
            crop_at_peak(root / r["rel_path"], m).save(out / "crops" / f"{stem}.png")
            w.writerow([r["image_id"], r["label"], r["defect_type"], f"{s:.6f}"])
    labels = np.array([r["label"] == "defect" for r in test])
    scores = np.array([s for _, s in maps])
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(labels, scores)
    print(f"{category:<12} 정상 {len(good):>4}장 뱅크, 테스트 {len(test):>4}장, 이미지 AUROC {auroc * 100:.1f}")
    return auroc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--index", type=Path, default=Path("data/index.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/mvtec_ad"))
    ap.add_argument("--out", type=Path, default=Path("data/patchcore"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with open(args.index, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    model = Backbone().to(args.device).eval()
    cats = CATEGORIES if args.all else [args.category]
    aurocs = []
    for c in cats:
        aurocs.append(run_category(c, args.root, args.out, rows, args.device, model))
    if len(cats) > 1:
        print(f"평균 이미지 AUROC {np.mean(aurocs) * 100:.1f}")
        with open(args.out / "auroc.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["category", "image_auroc"])
            w.writerows(zip(cats, [f"{a:.4f}" for a in aurocs]))


if __name__ == "__main__":
    main()
