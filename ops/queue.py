"""학습과 추론 작업 대기열을 만든다(ops/runtime/queue.json). 감독자(ops.supervisor)가 순서대로 실행한다.

작업 하나는 python -m 으로 실행할 인자 목록(argv), 완료를 판정하는 파일 목록(done),
재시도 때 덧붙일 인자(resume)로 적는다. 단계 이름을 여러 개 주면 그 순서대로 이어 붙인다.

단계
    stage1   사전학습 노출 탐침(4B), 4B zero-shot 5-fold(NF4, 결함 사전 포함 프롬프트)
    stage2   2B LoRA 5-fold, 입력 구성 네 가지(전체, crop 없음, 참조 없음, 검사 이미지만)
    stage2x  2B LoRA 5-fold, 정의 없는 사전(예산이 남을 때)

사용법:
    python -m ops.queue stage1 stage2 --out ops/runtime/queue.json
    python -m ops.queue stage1 stage2 --list
"""

import argparse
import json
from pathlib import Path

FOLDS = (1, 2, 3, 4, 5)
MODEL_2B = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_4B = "Qwen/Qwen3-VL-4B-Instruct"
# 입력 구성 이름과 src.vlm.build_sft 출력 폴더 접미사
VARIANTS_2B = {"full": "", "nocrop": "_nocrop", "noref": "_noref", "img": "_img"}
GEN_BATCH_4B = 4  # 4B NF4 생성은 배치 4에서 6.5GB였다. 더 큰 배치는 재지 않았다


def job(job_id: str, argv: list[str], done: list[str], resume: list[str] | None = None) -> dict:
    return {"id": job_id, "argv": argv, "done": done, "resume": resume or [], "status": "pending",
            "attempts": 0, "pid": None, "started": None, "finished": None, "error": None}


def stage1(gen_batch: int) -> list[dict]:
    jobs = [job("probe_4b",
                ["-m", "src.vlm.probe", "--model", MODEL_4B, "--out", "reports/tables/probe_4b.json"],
                ["reports/tables/probe_4b.json"], ["--resume"])]
    for k in FOLDS:
        out = f"runs/z4b/f{k}_predictions.jsonl"
        metrics = f"runs/z4b/f{k}_metrics.json"
        jobs.append(job(f"z4b_f{k}",
                        ["-m", "src.vlm.infer", "--data", f"data/sft/f{k}/eval.jsonl", "--model", MODEL_4B,
                         "--quant", "nf4", "--batch-size", str(GEN_BATCH_4B), "--out", out, "--metrics-out", metrics],
                        [out, metrics], ["--resume"]))
    return jobs


def train_job(job_id: str, data: str, out: str, model: str, gen_batch: int, quant: str | None = None) -> dict:
    argv = ["-m", "src.vlm.train_lora", "--data", data, "--model", model, "--out", out,
            "--gen-batch-size", str(gen_batch)]
    if quant:
        argv += ["--quant", quant]
    return job(job_id, argv, [f"{out}/final/adapter_config.json", f"{out}/eval_metrics.json"], ["--resume"])


def stage2(gen_batch: int) -> list[dict]:
    # fold를 바깥 루프에 두어 fold 1의 네 구성이 먼저 끝나고 이른 비교가 가능하게 한다
    return [train_job(f"2b_{name}_f{k}", f"data/sft/f{k}{suffix}", f"runs/2b_{name}/f{k}", MODEL_2B, gen_batch)
            for k in FOLDS for name, suffix in VARIANTS_2B.items()]


def stage2x(gen_batch: int) -> list[dict]:
    return [train_job(f"2b_nodef_f{k}", f"data/sft/f{k}_nodef", f"runs/2b_nodef/f{k}", MODEL_2B, gen_batch)
            for k in FOLDS]


def smoke(gen_batch: int) -> list[dict]:
    """감독자 경로 점검용 20 step 스모크 세 가지. 본 대기열을 돌리기 전에 설정 오류를 거르고 속도 설정을 고른다."""
    variants = {"smoke_2b": [], "smoke_2b_nockpt": ["--no-grad-ckpt"],
                "smoke_2b_nockpt_b2": ["--no-grad-ckpt", "--batch-size", "2"]}
    jobs = []
    for name, extra in variants.items():
        out = f"runs/smoke_sup/{name}"
        argv = ["-m", "src.vlm.train_lora", "--data", "data/sft/f1", "--model", MODEL_2B, "--out", out,
                "--smoke", "--gen-batch-size", str(gen_batch), *extra]
        jobs.append(job(name, argv, [f"{out}/eval_metrics.json"], ["--resume"]))
    return jobs


STAGES = {"smoke": smoke, "stage1": stage1, "stage2": stage2, "stage2x": stage2x}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=sorted(STAGES))
    ap.add_argument("--out", type=Path, default=Path("ops/runtime/queue.json"))
    ap.add_argument("--gen-batch", type=int, default=4, help="생성(추론) 배치 크기")
    ap.add_argument("--list", action="store_true", help="파일을 쓰지 않고 작업 ID만 보인다")
    args = ap.parse_args()

    jobs = []
    for s in args.stages:
        jobs.extend(STAGES[s](args.gen_batch))
    ids = [j["id"] for j in jobs]
    if len(ids) != len(set(ids)):
        raise SystemExit("작업 ID 중복")
    if args.list:
        for j in jobs:
            print(f"{j['id']:<18} python {' '.join(j['argv'])}")
        print(f"작업 {len(jobs)}개")
        return
    if args.out.is_file():
        # 이미 진행한 대기열의 상태(done, failed, attempts)는 보존하고 새 작업만 덧붙인다
        old = {j["id"]: j for j in json.loads(args.out.read_text(encoding="utf-8"))["jobs"]}
        jobs = [old.get(j["id"], j) for j in jobs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"jobs": jobs}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"작업 {len(jobs)}개 저장: {args.out}")


if __name__ == "__main__":
    main()
