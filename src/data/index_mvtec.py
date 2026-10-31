"""MVTec AD 폴더를 훑어 이미지 인덱스(data/index.csv)를 만든다.

폴더 규약(공식 아카이브 그대로):
    <root>/<category>/train/good/<id>.png
    <root>/<category>/test/<good|defect_type>/<id>.png
    <root>/<category>/ground_truth/<defect_type>/<id>_mask.png

인덱스에는 이미지 ID, 제품, 분할, 라벨, 결함 유형, 상대 경로, 마스크 경로, 해상도만 담는다.
이미지 자체는 저장소에 넣지 않으므로(CC BY-NC-SA 4.0) 이 파일만으로 어떤 이미지를 썼는지 재현할 수 있다.

사용법:
    python -m src.data.index_mvtec --root data/mvtec_ad --out data/index.csv
"""

import argparse
import csv
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}
FIELDS = ["image_id", "category", "kind", "split", "label", "defect_type", "rel_path", "mask_rel_path", "width", "height"]


def scan(root: Path) -> list[dict]:
    rows = []
    for category in CATEGORIES:
        cdir = root / category
        if not cdir.is_dir():
            raise FileNotFoundError(f"제품 폴더 없음: {cdir}")
        for split in ("train", "test"):
            for label_dir in sorted((cdir / split).iterdir()):
                if not label_dir.is_dir():
                    continue
                defect_type = label_dir.name
                label = "good" if defect_type == "good" else "defect"
                for img in sorted(label_dir.iterdir()):
                    if img.suffix.lower() not in IMAGE_EXTS:
                        continue
                    mask_rel = ""
                    if label == "defect":
                        mask = cdir / "ground_truth" / defect_type / f"{img.stem}_mask{img.suffix}"
                        if not mask.is_file():
                            raise FileNotFoundError(f"마스크 없음: {mask}")
                        mask_rel = mask.relative_to(root).as_posix()
                    with Image.open(img) as im:
                        width, height = im.size
                    rows.append({
                        "image_id": f"{category}/{split}/{defect_type}/{img.stem}",
                        "category": category,
                        "kind": "texture" if category in TEXTURES else "object",
                        "split": split,
                        "label": label,
                        "defect_type": "" if label == "good" else defect_type,
                        "rel_path": img.relative_to(root).as_posix(),
                        "mask_rel_path": mask_rel,
                        "width": width,
                        "height": height,
                    })
    return rows


def summarize(rows: list[dict]) -> str:
    lines = [f"{'제품':<12}{'train good':>11}{'test good':>10}{'test defect':>12}{'유형 수':>8}"]
    for category in CATEGORIES:
        sub = [r for r in rows if r["category"] == category]
        n_train = sum(r["split"] == "train" for r in sub)
        n_tg = sum(r["split"] == "test" and r["label"] == "good" for r in sub)
        n_td = sum(r["split"] == "test" and r["label"] == "defect" for r in sub)
        n_types = len({r["defect_type"] for r in sub if r["defect_type"]})
        lines.append(f"{category:<12}{n_train:>11}{n_tg:>10}{n_td:>12}{n_types:>8}")
    n_types_all = len({(r["category"], r["defect_type"]) for r in rows if r["defect_type"]})
    lines.append(f"{'합계':<12}{sum(r['split'] == 'train' for r in rows):>11}"
                 f"{sum(r['split'] == 'test' and r['label'] == 'good' for r in rows):>10}"
                 f"{sum(r['split'] == 'test' and r['label'] == 'defect' for r in rows):>12}{n_types_all:>8}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/mvtec_ad"))
    ap.add_argument("--out", type=Path, default=Path("data/index.csv"))
    args = ap.parse_args()

    rows = scan(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)}장 -> {args.out}".replace("->", "기록:"))
    print(summarize(rows))


if __name__ == "__main__":
    main()
