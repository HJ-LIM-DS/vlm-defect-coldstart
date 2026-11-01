"""LoRA 어댑터 또는 기본 모델로 예측을 생성해 JSONL로 저장한다(greedy).

입력 JSONL은 src.vlm.build_sft 가 만든 형식(messages, target, image_id, category, label, defect_type).
출력 행: image_id, category, label, defect_type, output(원문), pred_key, pred_name, reason, format_ok
지표 JSON은 --metrics-out 에 쓴다(기본은 출력 파일 이름의 _predictions 를 _metrics 로 바꾼 .json).

생성은 배치마다 파일에 바로 덧붙이므로 중단되어도 --resume 으로 이어 갈 수 있다.
HEARTBEAT_FILE 환경 변수가 있으면 배치마다 진행 상태를 그 파일에 쓴다(ops.supervisor 가 읽는다).

사용법:
    python -m src.vlm.infer --data data/sft/f1/eval.jsonl --adapter runs/f1_2b/final --out runs/f1_2b/eval_predictions.jsonl
    python -m src.vlm.infer --data data/sft/f1/eval.jsonl --model Qwen/Qwen3-VL-4B-Instruct --quant nf4 --out runs/z4b/f1_predictions.jsonl
"""

import argparse
import json
import os
import time
from pathlib import Path

# transformers.trainer 가 sklearn 을 거쳐 pandas 를 아주 깊은 import 사슬 끝에서 적재하면 Windows 메인 스레드
# 스택(1MB)이 넘칠 수 있다. 여기서 얕은 깊이로 먼저 적재한다. 같은 이유로 pyarrow 와 datasets 는 환경에 두지 않는다
# (torch 뒤에 pyarrow.dataset 을 적재하면 번들 DLL 충돌로 접근 위반이 난다).
import pandas  # noqa: F401
import torch
from PIL import Image

from src.data.lexicon import load_lexicon
from src.vlm.prompts import parse_output

IMAGE_PIXELS = 512 * 512  # 이미지당 약 256 토큰


def write_heartbeat(step: int, max_steps: int, phase: str, loss: float | None = None):
    """HEARTBEAT_FILE 환경 변수가 가리키는 파일에 진행 상태를 쓴다. 변수가 없으면 아무것도 하지 않는다.

    감독자는 이 파일의 시각이 오래 멈추면 프로세스가 죽은 것으로 보고 재시도한다.
    """
    path = os.environ.get("HEARTBEAT_FILE")
    if not path:
        return
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"time": time.time(), "step": step, "max_steps": max_steps,
                                   "phase": phase, "loss": loss}), encoding="utf-8")
        tmp.replace(p)
    except OSError:
        pass  # 감독자가 읽는 중이면 다음 하트비트에 쓴다


def load_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows[:limit] if limit else rows


def open_images(messages: list[dict]) -> list[Image.Image]:
    return [Image.open(c["image"]).convert("RGB") for c in messages[0]["content"] if c["type"] == "image"]


def load_processor(model_id: str):
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id)
    proc.image_processor.max_pixels = IMAGE_PIXELS
    proc.image_processor.min_pixels = IMAGE_PIXELS // 4
    return proc


def load_model(model_id: str, adapter: str | None, quant: str | None = None):
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    kw = {"dtype": torch.bfloat16, "attn_implementation": "sdpa", "device_map": {"": 0}}
    if quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"])
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return model.eval()


@torch.no_grad()
def generate(model, processor, rows: list[dict], batch_size: int = 4, max_new_tokens: int = 64, on_batch=None) -> list[str]:
    tok = processor.tokenizer
    outs = []
    for i in range(0, len(rows), batch_size):
        if on_batch is not None:
            on_batch(i, len(rows))
        batch = rows[i:i + batch_size]
        texts = [processor.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=False) for r in batch]
        imgs = [open_images(r["messages"]) for r in batch]
        enc = processor(text=texts, images=imgs, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for g in gen[:, enc.input_ids.shape[1]:]:
            outs.append(tok.decode(g, skip_special_tokens=True).strip())
    return outs


def to_record(r: dict, output: str, lex: dict) -> dict:
    p = parse_output(output, lex, r["category"])
    return {"image_id": r["image_id"], "category": r["category"], "label": r["label"],
            "defect_type": r["defect_type"], "output": output, "pred_key": p["key"], "pred_name": p["name"],
            "reason": p["reason"], "format_ok": p["format_ok"]}


def predict_to_file(model, processor, rows: list[dict], out_path: Path, batch_size: int = 4,
                    resume: bool = False, phase: str = "gen") -> dict:
    """배치마다 생성해 파일에 바로 덧붙인다. resume 이면 파일에 이미 있는 이미지는 건너뛴다."""
    lex = load_lexicon()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if resume and out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["image_id"]] = rec
    todo = [r for r in rows if r["image_id"] not in done]
    n_prev = len(done)  # 재개 전에 끝난 수. done 은 아래에서 계속 늘어나므로 미리 고정한다
    t0 = time.time()
    with open(out_path, "a" if done else "w", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            write_heartbeat(n_prev + i, len(rows), phase)
            batch = todo[i:i + batch_size]
            for r, o in zip(batch, generate(model, processor, batch, batch_size=batch_size)):
                done[r["image_id"]] = to_record(r, o, lex)
                f.write(json.dumps(done[r["image_id"]], ensure_ascii=False) + "\n")
            f.flush()
    sec = time.time() - t0
    recs = [done[r["image_id"]] for r in rows]
    # 재개로 순서가 섞였을 수 있으니 입력 순서대로 다시 쓴다
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    write_heartbeat(len(rows), len(rows), phase)
    n_defect = sum(r["label"] == "defect" for r in recs)
    n_correct = sum(r["label"] == "defect" and r["pred_key"] == r["defect_type"] for r in recs)
    return {"n": len(rows), "n_resumed": len(rows) - len(todo),
            "format_rate": sum(bool(r["format_ok"]) for r in recs) / max(len(rows), 1),
            "type_acc_image_mean": n_correct / max(n_defect, 1), "seconds_per_image": sec / max(len(todo), 1),
            "max_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None}


def default_metrics_path(out: Path) -> Path:
    return out.with_name(out.stem.replace("_predictions", "") + "_metrics.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--metrics-out", type=Path, default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--quant", choices=["nf4"], default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="출력 파일에 이미 있는 이미지는 건너뛴다")
    args = ap.parse_args()

    rows = load_rows(args.data, args.limit)
    processor = load_processor(args.model)
    model = load_model(args.model, args.adapter, args.quant)
    m = predict_to_file(model, processor, rows, args.out, args.batch_size, resume=args.resume)
    m.update({"model": args.model, "adapter": args.adapter, "quant": args.quant, "data": str(args.data)})
    metrics_path = args.metrics_out or default_metrics_path(args.out)
    metrics_path.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(m, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
