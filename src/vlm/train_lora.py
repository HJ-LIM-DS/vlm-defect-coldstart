"""Qwen3-VL에 LoRA(bf16) 또는 QLoRA(NF4)를 붙여 학습한다. 손실은 정답 토큰에만 준다.

입력: src.vlm.build_sft 가 만든 폴더(train.jsonl, val.jsonl, eval.jsonl)
출력(--out 폴더):
    final/              어댑터(PEFT 형식)
    train_log.json      설정, 손실 이력, 학습 시간, 최대 VRAM
    val_predictions.jsonl, val_metrics.json      학습 제품 안 검증 분할의 생성 결과
    eval_predictions.jsonl, eval_metrics.json    평가 제품(fold)의 생성 결과

메모리를 아끼는 장치: 왼쪽 패딩으로 정답을 시퀀스 끝에 모으고 마지막 k+1 위치의 logits만 계산한다.
비전 인코더는 동결한다(LoRA 대상은 언어 모델의 attention과 MLP projection).

사용법:
    python -m src.vlm.train_lora --data data/sft/f1 --model Qwen/Qwen3-VL-2B-Instruct --out runs/f1_2b
    python -m src.vlm.train_lora --data data/sft/f1 --model Qwen/Qwen3-VL-4B-Instruct --quant nf4 --out runs/f1_4b
    python -m src.vlm.train_lora ... --smoke     # 20 step만 돌려 설정 오류와 OOM을 먼저 거른다
"""

import argparse
import contextlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, SequentialSampler

from src.vlm.infer import IMAGE_PIXELS, load_processor, load_rows, open_images, predict_to_file, write_heartbeat
from src.vlm.infer import load_model as load_trained_model

LM_TARGETS = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


class SftDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {"images": open_images(r["messages"]), "messages": r["messages"], "target": r["target"]}


class Collator:
    """표본마다 프롬프트를 토큰화하고 정답 토큰을 이어 붙인 뒤 왼쪽 패딩한다."""

    def __init__(self, processor):
        self.p = processor
        self.tok = processor.tokenizer
        self.end_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.pad_id = self.tok.pad_token_id

    def encode(self, ex):
        text = self.p.apply_chat_template(ex["messages"], add_generation_prompt=True, tokenize=False)
        enc = self.p(text=[text], images=ex["images"], return_tensors="pt")
        tgt = self.tok(ex["target"], add_special_tokens=False)["input_ids"] + [self.end_id]
        return enc, tgt

    def __call__(self, batch):
        encs = [self.encode(ex) for ex in batch]
        lens = [e.input_ids.shape[1] + len(t) for e, t in encs]
        L, B = max(lens), len(batch)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        att = torch.zeros((B, L), dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        mm = torch.zeros((B, L), dtype=torch.long)  # 이미지 토큰 표시(위치 인코딩용). 정답과 패딩은 텍스트(0)
        for b, ((e, t), n) in enumerate(zip(encs, lens)):
            ids[b, L - n:] = torch.cat([e.input_ids[0], torch.tensor(t)])
            att[b, L - n:] = 1
            labels[b, L - len(t):] = torch.tensor(t)
            mm[b, L - n:L - len(t)] = e["mm_token_type_ids"][0]
        return {
            "input_ids": ids, "attention_mask": att, "labels": labels, "mm_token_type_ids": mm,
            "pixel_values": torch.cat([e.pixel_values for e, _ in encs]),
            "image_grid_thw": torch.cat([e.image_grid_thw for e, _ in encs]),
            "n_keep": max(len(t) for _, t in encs),
        }


@contextlib.contextmanager
def trusted_checkpoint(checkpoint: str, output_dir: str):
    """같은 run이 출력 폴더 안에 저장한 체크포인트를 재개할 때만 torch.load 전체 로드를 허용한다.

    torch 2.5에서는 transformers가 optimizer와 RNG 상태 로드를 막는다(weights_only 허용 목록 문제).
    읽는 파일이 이 run의 출력 폴더 안에 있을 때만, 그 호출 동안만 검사를 끈다.
    """
    import transformers.trainer as tt

    ckpt = Path(checkpoint).resolve()
    if Path(output_dir).resolve() not in ckpt.parents:
        raise ValueError(f"출력 폴더 밖 체크포인트: {ckpt}")
    original_check, original_load = tt.check_torch_load_is_safe, torch.load

    def trusted_load(f, *a, **kw):
        if Path(f).resolve().parent != ckpt:
            raise ValueError(f"체크포인트 밖 파일 로드 시도: {f}")
        kw["weights_only"] = False
        return original_load(f, *a, **kw)

    tt.check_torch_load_is_safe = lambda: None
    torch.load = trusted_load
    try:
        yield
    finally:
        tt.check_torch_load_is_safe = original_check
        torch.load = original_load


def make_trainer_cls():
    from transformers import Trainer

    class AnswerOnlyTrainer(Trainer):
        def __init__(self, *a, smoke: bool = False, **kw):
            super().__init__(*a, **kw)
            self.smoke = smoke
            self.model_accepts_loss_kwargs = True  # accumulation 창 전체 정답 토큰 수로 직접 정규화한다

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            k = int(inputs.pop("n_keep"))
            out = model(**inputs, logits_to_keep=k + 1)
            logits = out.logits[:, :-1, :].float()
            tgt = labels[:, -k:].to(logits.device)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100, reduction="sum")
            denom = num_items_in_batch if num_items_in_batch is not None else (tgt != -100).sum()
            loss = loss / denom
            return (loss, out) if return_outputs else loss

        def _load_optimizer_and_scheduler(self, checkpoint):
            if checkpoint is None:
                return super()._load_optimizer_and_scheduler(checkpoint)
            with trusted_checkpoint(checkpoint, self.args.output_dir):
                return super()._load_optimizer_and_scheduler(checkpoint)

        def _load_rng_state(self, checkpoint):
            if checkpoint is None:
                return super()._load_rng_state(checkpoint)
            with trusted_checkpoint(checkpoint, self.args.output_dir):
                return super()._load_rng_state(checkpoint)

        def _get_train_sampler(self, *a, **kw):
            if self.smoke:  # 가장 긴 표본부터: OOM을 첫 step에서 드러낸다
                return SequentialSampler(self.train_dataset)
            return super()._get_train_sampler(*a, **kw)

    return AnswerOnlyTrainer


def load_model(model_id: str, quant: str | None, lora_r: int, lora_alpha: int, grad_ckpt: bool = True):
    """gradient checkpointing은 메모리를 아끼는 대신 느리다. 2B처럼 메모리가 남는 설정에서는 끈다."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    kw = {"dtype": torch.bfloat16, "attn_implementation": "sdpa", "device_map": {"": 0}}
    if quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"])
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
    if quant == "nf4":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=grad_ckpt)
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                                             target_modules=LM_TARGETS, task_type="CAUSAL_LM"))
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    return model


class HeartbeatCallback:
    """학습 step마다 진행 상태를 하트비트 파일에 쓴다(src.vlm.infer.write_heartbeat, 감독자가 읽는다)."""

    def __init__(self):
        from transformers import TrainerCallback

        class _CB(TrainerCallback):
            def on_log(cb, args, state, control, logs=None, **kw):
                write_heartbeat(state.global_step, state.max_steps, "train", (logs or {}).get("loss"))

            def on_step_end(cb, args, state, control, **kw):
                write_heartbeat(state.global_step, state.max_steps, "train")

        self.cb = _CB()


def latest_checkpoint(out_dir: Path) -> str | None:
    cks = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    for ck in reversed(cks):
        if (ck / "trainer_state.json").is_file() and (ck / "adapter_model.safetensors").is_file():
            return str(ck)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--quant", choices=["nf4"], default=None)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--gen-batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true", help="출력 폴더의 최신 체크포인트에서 재개")
    ap.add_argument("--save-steps", type=int, default=50, help="체크포인트 저장 간격(step). 중단 시 잃는 시간을 정한다")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="gradient checkpointing을 끈다(메모리가 남을 때 더 빠르다)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    final = args.out / "final"
    trained = (final / "adapter_config.json").is_file() and not args.smoke
    if trained and (args.out / "eval_metrics.json").is_file():
        print("이미 끝난 run:", args.out)
        return
    train_rows, val_rows, eval_rows = (load_rows(args.data / f"{n}.jsonl") for n in ("train", "val", "eval"))
    if args.smoke:
        train_rows, val_rows, eval_rows = train_rows[:40], val_rows[:8], eval_rows[:8]

    processor = load_processor(args.model)
    if trained:
        # 학습은 끝났고 생성 중에 중단된 run: 어댑터를 읽어 생성만 이어 간다
        print("학습이 끝난 run, 생성만 수행:", args.out)
        model = load_trained_model(args.model, str(final), args.quant)
    else:
        model = train(args, processor, train_rows)

    for name, rows in (("val", val_rows), ("eval", eval_rows)):
        m = predict_to_file(model, processor, rows, args.out / f"{name}_predictions.jsonl", args.gen_batch_size,
                            resume=args.resume, phase=f"gen_{name}")
        (args.out / f"{name}_metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        print(name, json.dumps(m, ensure_ascii=False))
    print(f"끝: {args.out}")


def train(args, processor, train_rows: list[dict]):
    """LoRA를 학습해 final/ 에 저장하고, 생성에 쓸 수 있게 학습 메모리를 비운 모델을 돌려준다."""
    from transformers import TrainingArguments

    model = load_model(args.model, args.quant, args.lora_r, args.lora_alpha, grad_ckpt=not args.no_grad_ckpt)
    model.print_trainable_parameters()

    accum = 1 if args.smoke else args.grad_accum
    total_steps = 20 if args.smoke else int(-(-len(train_rows) // (args.batch_size * accum)) * args.epochs)
    targs = TrainingArguments(
        output_dir=str(args.out), seed=args.seed,
        num_train_epochs=args.epochs, max_steps=20 if args.smoke else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=max(1, int(0.03 * total_steps)), weight_decay=0.0,
        bf16=True, logging_steps=10, save_strategy="no" if args.smoke else "steps", save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[], remove_unused_columns=False, dataloader_num_workers=0, gradient_checkpointing=False,
        optim="paged_adamw_8bit" if args.quant == "nf4" else "adamw_torch",
    )
    trainer = make_trainer_cls()(
        model=model, args=targs, train_dataset=SftDataset(train_rows), data_collator=Collator(processor),
        callbacks=[HeartbeatCallback().cb], smoke=args.smoke,
    )
    resume = latest_checkpoint(args.out) if args.resume else None
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    train_sec = time.time() - t0
    max_vram = torch.cuda.max_memory_allocated() / 1e9

    final = args.out / "final"
    model.save_pretrained(final)
    processor.save_pretrained(final)
    log = {
        "model": args.model, "quant": args.quant, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "grad_ckpt": not args.no_grad_ckpt, "save_steps": args.save_steps,
        "image_pixels": IMAGE_PIXELS, "seed": args.seed, "smoke": args.smoke, "resumed_from": resume,
        "n_train": len(train_rows), "train_seconds": round(train_sec, 1), "max_vram_gb": round(max_vram, 2),
        "loss_history": [{"step": h["step"], "loss": h["loss"]} for h in trainer.state.log_history if "loss" in h],
    }
    (args.out / "train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"학습 끝: {train_sec / 3600:.2f}시간, 최대 VRAM {max_vram:.2f}GB")

    # 생성 전에 학습용 메모리를 비운다(VRAM 경합으로 생성이 느려지는 것을 막는다)
    model.config.use_cache = True
    model.eval()
    del trainer
    torch.cuda.empty_cache()
    return model


if __name__ == "__main__":
    main()
