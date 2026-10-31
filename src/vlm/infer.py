"""LoRA 어댑터 또는 기본 모델로 예측을 생성해 JSONL로 저장한다(greedy).

입력 JSONL은 src.vlm.build_sft 가 만든 형식(messages, target, image_id, category, label, defect_type).
출력 행: image_id, category, label, defect_type, output(원문), pred_key, pred_name, reason, format_ok

사용법:
    python -m src.vlm.infer --data data/sft/f1/eval.jsonl --adapter runs/f1_2b/final --out runs/f1_2b/eval_predictions.jsonl
    python -m src.vlm.infer --data data/sft/f1/eval.jsonl --model Qwen/Qwen3-VL-4B-Instruct --out runs/zeroshot_f1.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from src.data.lexicon import load_lexicon
from src.vlm.prompts import parse_output

IMAGE_PIXELS = 512 * 512  # 이미지당 약 256 토큰


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


def predict_to_file(model, processor, rows: list[dict], out_path: Path, batch_size: int = 4, on_batch=None) -> dict:
    lex = load_lexicon()
    t0 = time.time()
    outputs = generate(model, processor, rows, batch_size=batch_size, on_batch=on_batch)
    sec = time.time() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_correct = n_defect = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outputs):
            p = parse_output(o, lex, r["category"])
            n_ok += p["format_ok"]
            if r["label"] == "defect":
                n_defect += 1
                n_correct += p["key"] == r["defect_type"]
            f.write(json.dumps({
                "image_id": r["image_id"], "category": r["category"], "label": r["label"],
                "defect_type": r["defect_type"], "output": o, "pred_key": p["key"], "pred_name": p["name"],
                "reason": p["reason"], "format_ok": p["format_ok"],
            }, ensure_ascii=False) + "\n")
    return {"n": len(rows), "format_rate": n_ok / max(len(rows), 1),
            "type_acc_image_mean": n_correct / max(n_defect, 1), "seconds_per_image": sec / max(len(rows), 1),
            "max_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--quant", choices=["nf4"], default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = load_rows(args.data, args.limit)
    processor = load_processor(args.model)
    model = load_model(args.model, args.adapter, args.quant)
    m = predict_to_file(model, processor, rows, args.out, args.batch_size)
    print(json.dumps(m, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
