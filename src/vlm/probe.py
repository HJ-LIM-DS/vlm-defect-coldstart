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

from src.vlm.infer import generate, load_model, load_processor, write_heartbeat
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
    ap.add_argument("--resume", action="store_true", help="부분 파일에 이미 있는 이미지는 건너뛴다")
    args = ap.parse_args()

    with open(args.index, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["label"] == "defect"]
    rows = rows[:args.limit] if args.limit else rows
    # 배치마다 부분 파일에 덧붙여, 중단되면 --resume 으로 이어 간다
    partial = args.out.with_suffix(".partial.jsonl")
    done: dict[str, str] = {}
    if args.resume and partial.is_file():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["image_id"]] = rec["output"]
    todo = [r for r in rows if r["image_id"] not in done]
    n_prev = len(done)  # 재개 전에 끝난 수. done 은 아래에서 계속 늘어나므로 미리 고정한다
    processor = load_processor(args.model)
    model = load_model(args.model, None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(partial, "a" if done else "w", encoding="utf-8") as f:
        for i in range(0, len(todo), args.batch_size):
            write_heartbeat(n_prev + i, len(rows), "probe")
            batch = todo[i:i + args.batch_size]
            items = [{"messages": build_probe_messages((args.root / r["rel_path"]).as_posix())} for r in batch]
            for r, o in zip(batch, generate(model, processor, items, batch_size=args.batch_size, max_new_tokens=16)):
                done[r["image_id"]] = o
                f.write(json.dumps({"image_id": r["image_id"], "output": o}, ensure_ascii=False) + "\n")
            f.flush()
    outputs = [done[r["image_id"]] for r in rows]

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
