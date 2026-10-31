"""사전학습 노출 탐침: 결함 사전 없이 결함 종류를 영어로 묻고 폴더 이름을 그대로 맞히는 비율을 잰다.

모델이 MVTec AD를 사전학습에서 봤다면 유형 이름(예: "broken_large")을 사전 없이도 말할 수 있다.
이 비율은 해석을 위한 참고값이며 가설 판정에는 쓰지 않는다(docs/preregistration.md 8절).

일치 규칙: 출력을 소문자로 바꾸고 공백과 밑줄을 하나로 본 뒤, 폴더 이름(밑줄을 공백으로)이 출력에 포함되면 일치.

사용법:
    python -m src.vlm.probe --model Qwen/Qwen3-VL-4B-Instruct --out reports/tables/probe_4b.json
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from src.vlm.infer import generate, load_model, load_processor
from src.vlm.prompts import build_probe_messages


def normalize(s: str) -> str:
    return re.sub(r"[\s_]+", " ", s.lower()).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--index", type=Path, default=Path("data/index.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/mvtec_ad"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(args.index, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["label"] == "defect"]
    rows = rows[:args.limit] if args.limit else rows
    items = [{"messages": build_probe_messages((args.root / r["rel_path"]).as_posix())} for r in rows]
    processor = load_processor(args.model)
    model = load_model(args.model, None)
    outputs = generate(model, processor, items, batch_size=args.batch_size, max_new_tokens=16)

    hits = defaultdict(list)
    records = []
    for r, o in zip(rows, outputs):
        hit = normalize(r["defect_type"]) in normalize(o)
        hits[r["category"]].append(hit)
        records.append({"image_id": r["image_id"], "defect_type": r["defect_type"], "output": o, "hit": hit})
    summary = {
        "model": args.model, "n": len(rows),
        "exact_name_rate": sum(sum(v) for v in hits.values()) / max(len(rows), 1),
        "per_product": {k: sum(v) / len(v) for k, v in sorted(hits.items())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
