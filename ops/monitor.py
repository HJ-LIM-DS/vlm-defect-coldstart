"""감독자 진행 상태를 터미널에 상시 표시한다(별도 창에 띄워 둔다). 기본 30초마다 갱신한다.

표시: 현재 작업과 단계(학습, 검증 생성, 평가 생성, 탐침)별 진행 막대, 손실, 예상 남은 시간, 마지막 하트비트 경과,
GPU 메모리와 사용률, 대기열 요약과 작업별 소요 시간, 현재 작업 로그의 마지막 줄.
ops/runtime/status.json 과 하트비트 파일을 읽기만 한다. 한 번에 하나만 뜬다(락 파일).

사용법:
    python -m ops.monitor
    python -m ops.monitor --interval 10
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from filelock import FileLock, Timeout
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

RUNTIME = Path("ops/runtime")
PHASES = {"train": "학습", "gen_val": "검증 생성", "gen_eval": "평가 생성", "gen": "생성", "probe": "탐침"}


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def gpu_line() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        used, total, util, temp = [x.strip() for x in out.strip().split(",")]
        return f"GPU 메모리 {int(used) / 1024:.1f}/{int(total) / 1024:.1f}GB, 사용률 {util}%, 온도 {temp}C"
    except Exception:
        return "GPU 상태 읽기 실패"


def tail(path: Path, n: int = 6) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            lines = f.read().decode("utf-8", errors="replace").replace("\r", "\n").splitlines()
        lines = [x for x in lines if x.strip() and "Loading weights" not in x]
        return "\n".join(lines[-n:])
    except OSError:
        return ""


def fmt_minutes(sec: float) -> str:
    if sec < 0 or sec != sec:
        return "?"
    return f"{sec / 3600:.1f}시간" if sec >= 3600 else f"{sec / 60:.0f}분"


def elapsed(started: str | None, finished: str | None) -> str:
    if not started:
        return ""
    try:
        t0 = datetime.strptime(started, "%Y-%m-%d %H:%M:%S")
        t1 = datetime.strptime(finished, "%Y-%m-%d %H:%M:%S") if finished else datetime.now()
        return fmt_minutes((t1 - t0).total_seconds())
    except ValueError:
        return ""


class Speed:
    """같은 작업, 같은 단계 안에서 처음 본 하트비트와 비교해 속도와 남은 시간을 추정한다."""

    def __init__(self):
        self.key, self.t0, self.s0 = None, None, None

    def eta(self, job: str, hb: dict) -> str:
        key = (job, hb.get("phase"))
        if key != self.key or hb["step"] < (self.s0 or 0):
            self.key, self.t0, self.s0 = key, hb["time"], hb["step"]
            return "예상 남은 시간 계산 중"
        dt, ds = hb["time"] - self.t0, hb["step"] - self.s0
        if dt <= 0 or ds <= 0:
            return "예상 남은 시간 계산 중"
        remain = ((hb.get("max_steps") or 0) - hb["step"]) / (ds / dt)
        return f"속도 {ds / dt * 60:.1f} step/분, 예상 남은 {fmt_minutes(remain)}"


def render(status: dict | None, speed: Speed) -> Group:
    if status is None:
        return Group(Panel("status.json 없음. 감독자 시작 대기", title="감독자"))
    age = time.time() - status["time"]
    counts = status.get("counts", {})
    head = (f"감독자 pid {status.get('supervisor_pid')}, 상태 갱신 {age:.0f}초 전  |  "
            f"완료 {counts.get('done', 0)}  대기 {counts.get('pending', 0)}  실행 {counts.get('running', 0)}  실패 {counts.get('failed', 0)}")
    if age > 180:
        head += "  |  감독자 응답 없음(작업 스케줄러가 30분 안에 되살린다)"
    if status.get("note"):
        head += f"  |  {status['note']}"

    cur, hb = status.get("current"), status.get("heartbeat")
    progress = Progress(TextColumn("{task.description}"), BarColumn(bar_width=40),
                        TextColumn("{task.completed}/{task.total}"), TextColumn("{task.percentage:>5.1f}%"))
    lines = []
    if cur:
        if hb:
            hb_age = time.time() - hb["time"]
            phase = PHASES.get(hb.get("phase"), hb.get("phase"))
            progress.add_task(f"{cur} [{phase}]", total=max(hb.get("max_steps") or 1, 1), completed=hb.get("step") or 0)
            loss = f"손실 {hb['loss']:.4f}  " if isinstance(hb.get("loss"), (int, float)) else ""
            warn = "  (정지 의심)" if hb_age > status.get("stall_seconds", 1200) * 0.5 else ""
            lines.append(f"{loss}하트비트 {hb_age:.0f}초 전{warn}  |  {speed.eta(cur, hb)}")
        else:
            progress.add_task(f"{cur} [모델 로딩]", total=1, completed=0)
        lines.append(f"시작 {status.get('started')}, 경과 {elapsed(status.get('started'), None)}, 시도 {status.get('attempt')}, pid {status.get('pid')}")
    else:
        lines.append("실행 중인 작업 없음")
    lines.append(gpu_line())

    table = Table(show_header=True, header_style="bold", expand=False)
    for col in ("작업", "상태", "시도", "시작", "소요"):
        table.add_column(col)
    for j in status.get("queue", []):
        table.add_row(j["id"], j["status"], str(j["attempts"]), j.get("started") or "", elapsed(j.get("started"), j.get("finished")))

    parts = [Panel(head, title="감독자"), Panel(Group(progress, *lines), title="현재 작업")]
    if cur:
        parts.append(Panel(tail(RUNTIME / "logs" / f"{cur}.log"), title=f"로그 {cur}"))
    parts.append(Panel(table, title="대기열"))
    return Group(*parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=30.0)
    args = ap.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = FileLock(RUNTIME / "monitor.lock")
    try:
        lock.acquire(timeout=0)
    except Timeout:
        print("모니터가 이미 떠 있음")
        sys.exit(2)
    pid_file = RUNTIME / "monitor.pid"
    pid_file.write_text(str(os.getpid()), encoding="utf-8")
    speed = Speed()
    try:
        with Live(render(read_json(RUNTIME / "status.json"), speed), refresh_per_second=1, screen=False) as live:
            while True:
                time.sleep(args.interval)
                live.update(render(read_json(RUNTIME / "status.json"), speed))
    except KeyboardInterrupt:
        pass
    finally:
        pid_file.unlink(missing_ok=True)
        lock.release()


if __name__ == "__main__":
    main()
