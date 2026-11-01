"""학습이 모두 끝난 뒤 환경을 학습 전 상태로 되돌린다.

- 대기열의 학습 작업 출력 폴더에서 중간 체크포인트(checkpoint-*)를 지운다. 최종 어댑터(final/)와 예측, 지표, 로그는 남긴다.
- 작업 스케줄러의 감독자 되살리기 작업(ops/schedule.ps1 로 등록한 것)을 해제하고 모니터 창을 닫는다.
- 하트비트 파일, 상태 파일, pid 파일, 락 파일을 지운다. 감독자 로그와 작업 로그, 완료 요약은 기록으로 남긴다.
- 대기열에 failed 작업이 있으면 --force 없이는 아무것도 지우지 않는다(실패 원인 확인을 위해 체크포인트를 남긴다).

사용법:
    python -m ops.cleanup --dry-run
    python -m ops.cleanup
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

RUNTIME = Path("ops/runtime")
TASK_NAME = "vlm-defect-coldstart-supervisor"


def unschedule(dry_run: bool):
    if sys.platform != "win32":
        return
    exists = subprocess.run(["schtasks", "/Query", "/TN", TASK_NAME], capture_output=True).returncode == 0
    if not exists:
        print("예약 작업 없음")
        return
    print(f"{'(미리 보기) ' if dry_run else ''}예약 작업 해제 {TASK_NAME}")
    if not dry_run:
        subprocess.run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"], check=False)


def stop_monitor(dry_run: bool):
    pid_file = RUNTIME / "monitor.pid"
    if not pid_file.is_file():
        return
    try:
        import psutil

        p = psutil.Process(int(pid_file.read_text().strip()))
        if "python" in p.name().lower():
            print(f"{'(미리 보기) ' if dry_run else ''}모니터 종료 pid {p.pid}")
            if not dry_run:
                p.kill()
    except Exception:
        pass


def train_out_dirs(queue: dict) -> list[Path]:
    outs = []
    for j in queue["jobs"]:
        argv = j["argv"]
        if "src.vlm.train_lora" in argv and "--out" in argv:
            outs.append(Path(argv[argv.index("--out") + 1]))
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", type=Path, default=RUNTIME / "queue.json")
    ap.add_argument("--force", action="store_true", help="실패한 작업이 있어도 정리한다")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    failed = [j["id"] for j in queue["jobs"] if j["status"] == "failed"]
    unfinished = [j["id"] for j in queue["jobs"] if j["status"] not in ("done", "failed")]
    if unfinished:
        raise SystemExit(f"끝나지 않은 작업이 있음: {unfinished}")
    if failed and not args.force:
        raise SystemExit(f"실패한 작업이 있음(--force 로 강제): {failed}")

    unschedule(args.dry_run)
    stop_monitor(args.dry_run)
    targets = []
    for out in train_out_dirs(queue):
        targets.extend(sorted(out.glob("checkpoint-*")))
    targets.extend(sorted((RUNTIME / "hb").glob("*.json")))
    for name in ("status.json", "status.tmp", "supervisor.lock", "supervisor.pid", "monitor.lock", "monitor.pid"):
        if (RUNTIME / name).exists():
            targets.append(RUNTIME / name)

    freed = 0
    for t in targets:
        size = sum(p.stat().st_size for p in t.rglob("*") if p.is_file()) if t.is_dir() else t.stat().st_size
        freed += size
        print(f"{'(미리 보기) ' if args.dry_run else ''}삭제 {t} ({size / 1e9:.2f}GB)")
        if not args.dry_run:
            shutil.rmtree(t) if t.is_dir() else t.unlink()
    print(f"{'예상 ' if args.dry_run else ''}확보 용량 {freed / 1e9:.2f}GB, 항목 {len(targets)}개")


if __name__ == "__main__":
    main()
