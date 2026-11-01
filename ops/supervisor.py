"""대기열(ops/runtime/queue.json)의 작업을 하나씩 별도 프로세스로 실행하고 감시한다.

- 작업 프로세스는 콘솔에서 분리해 띄우고(감독자 창이 닫혀도 작업은 계속 돈다) HEARTBEAT_FILE 환경 변수가
  가리키는 파일로 진행 상태를 받는다. 하트비트가 --stall-minutes 동안 멈추면 프로세스를 끝내고 재시도한다.
  모델 로딩 중에는 하트비트가 없으므로 시작 직후에도 같은 유예를 준다.
- 감독자가 다시 시작됐을 때 이전 감독자가 띄운 작업이 아직 돌고 있으면 그 프로세스를 이어서 감시한다(입양).
  죽어 있으면 체크포인트에서 재개한다. 작업 스케줄러가 감독자를 주기적으로 되살리므로 어느 창이 닫혀도 학습이 이어진다.
- 재시도는 작업마다 --max-attempts 번까지. 그 안에 끝나지 않으면 failed 로 표시하고 다음 작업으로 넘어간다.
  GPU를 놀리지 않기 위해서다. 재시도할 때는 작업의 resume 인자(예: --resume)를 덧붙인다.
- 완료 판정은 작업의 done 파일이 모두 있는지로 한다. 이미 있으면 실행하지 않고 done 으로 표시한다.
- 한 번에 한 감독자만 돈다(락 파일). 진행 상태는 ops/runtime/status.json 에 쓰고 ops.monitor 가 읽는다.
- 모든 작업이 끝나면 ops/runtime/supervisor_done.json 에 요약을 쓰고 종료한다.

사용법:
    python -m ops.supervisor --queue ops/runtime/queue.json
    python -m ops.supervisor --queue ops/runtime/queue.json --python E:/venvs/x/Scripts/python.exe
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
from filelock import FileLock, Timeout

RUNTIME = Path("ops/runtime")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(RUNTIME / "supervisor.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_json(path: Path, obj: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_done(job: dict) -> bool:
    return all(Path(p).is_file() for p in job["done"])


def job_out(job: dict) -> str:
    argv = job["argv"]
    return argv[argv.index("--out") + 1] if "--out" in argv else job["id"]


class Adopted:
    """이전 감독자가 띄운 작업 프로세스. Popen 처럼 poll 과 kill 을 제공한다."""

    def __init__(self, proc: psutil.Process):
        self.proc = proc
        self.pid = proc.pid
        self.returncode = None

    def poll(self):
        if self.proc.is_running() and self.proc.status() != psutil.STATUS_ZOMBIE:
            return None
        if self.returncode is None:
            try:
                self.returncode = self.proc.wait(timeout=0)
            except Exception:
                self.returncode = -1
        return self.returncode

    def kill(self):
        self.proc.kill()

    def wait(self, timeout=None):
        return self.proc.wait(timeout=timeout)


def alive_job_process(job: dict) -> Adopted | None:
    """queue.json 에 적힌 pid 가 아직 이 작업을 돌리고 있으면 돌려준다(pid 재사용을 명령줄로 가린다)."""
    pid = job.get("pid")
    if not pid:
        return None
    try:
        p = psutil.Process(pid)
        cmd = " ".join(p.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    if "python" in p.name().lower() and job_out(job) in cmd:
        return Adopted(p)
    return None


class Supervisor:
    def __init__(self, queue_path: Path, python: str, stall_minutes: float, max_attempts: int, poll: float):
        self.queue_path = queue_path
        self.python = python
        self.stall = stall_minutes * 60
        self.max_attempts = max_attempts
        self.poll = poll
        self.queue = read_json(queue_path)
        if not self.queue:
            raise SystemExit(f"대기열 없음: {queue_path}")
        (RUNTIME / "hb").mkdir(parents=True, exist_ok=True)
        (RUNTIME / "logs").mkdir(parents=True, exist_ok=True)

    def save(self):
        write_json(self.queue_path, self.queue)

    def counts(self) -> dict:
        c = {}
        for j in self.queue["jobs"]:
            c[j["status"]] = c.get(j["status"], 0) + 1
        return c

    def status(self, job: dict | None, hb: dict | None, note: str = ""):
        write_json(RUNTIME / "status.json", {
            "time": time.time(), "supervisor_pid": os.getpid(), "stall_seconds": self.stall,
            "current": job["id"] if job else None, "pid": job["pid"] if job else None,
            "started": job["started"] if job else None, "attempt": job["attempts"] + 1 if job else None,
            "heartbeat": hb, "note": note, "counts": self.counts(),
            "queue": [{"id": j["id"], "status": j["status"], "attempts": j["attempts"],
                       "started": j["started"], "finished": j["finished"]} for j in self.queue["jobs"]],
        })

    def watch(self, job: dict, proc, hb_path: Path, lf) -> str:
        """프로세스가 끝나거나 하트비트가 멈출 때까지 지켜본다. 결과: done, stalled, exit<code>."""
        t_start = time.time()
        while True:
            code = proc.poll()
            hb = read_json(hb_path)
            last = hb["time"] if hb and "time" in hb else t_start
            self.status(job, hb)
            if code is not None:
                result = "done" if is_done(job) else f"exit{code}"
                break
            if time.time() - max(last, t_start) > self.stall:
                log(f"{job['id']} 하트비트 정지 {self.stall / 60:.0f}분 초과, 프로세스 종료")
                proc.kill()
                try:
                    proc.wait(timeout=60)
                except Exception:
                    pass
                result = "stalled"
                break
            time.sleep(self.poll)
        lf.write(f"===== [{now()}] 결과 {result}, 종료 코드 {proc.returncode}, 감시 {(time.time() - t_start) / 60:.1f}분\n")
        lf.flush()
        return result

    def run_once(self, job: dict) -> str:
        hb_path = RUNTIME / "hb" / f"{job['id']}.json"
        if hb_path.is_file():
            hb_path.unlink()
        # faulthandler: 네이티브 크래시(접근 위반)에서도 파이썬 스택을 로그에 남긴다
        argv = [self.python, "-X", "faulthandler", *job["argv"]] + (job["resume"] if job["attempts"] > 0 else [])
        env = dict(os.environ, HEARTBEAT_FILE=str(hb_path), PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                   PYTHONUNBUFFERED="1")
        with open(RUNTIME / "logs" / f"{job['id']}.log", "a", encoding="utf-8") as lf:
            lf.write(f"\n===== [{now()}] 시도 {job['attempts'] + 1}: {' '.join(argv)}\n")
            lf.flush()
            # 콘솔에서 분리: 감독자 창이 닫혀도 작업은 살아 있고, 다음 감독자가 입양한다
            flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS) if sys.platform == "win32" else 0
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, env=env,
                                    creationflags=flags)
            job["pid"], job["status"], job["started"] = proc.pid, "running", now()
            self.save()
            log(f"{job['id']} 시작 (pid {proc.pid}, 시도 {job['attempts'] + 1})")
            return self.watch(job, proc, hb_path, lf)

    def adopt(self, job: dict, proc: Adopted) -> str:
        with open(RUNTIME / "logs" / f"{job['id']}.log", "a", encoding="utf-8") as lf:
            lf.write(f"\n===== [{now()}] 감독자 재시작, 돌고 있는 pid {proc.pid} 를 이어서 감시\n")
            lf.flush()
            log(f"{job['id']} 입양 (pid {proc.pid})")
            return self.watch(job, proc, RUNTIME / "hb" / f"{job['id']}.json", lf)

    def run_job(self, job: dict):
        adopted = alive_job_process(job) if job["status"] == "running" else None
        if job["status"] == "running" and adopted is None:
            job["attempts"] = max(job["attempts"], 1)  # 이전 감독자와 함께 죽은 작업은 재시도로 취급한다
        while True:
            if adopted is not None:
                result = self.adopt(job, adopted)
                adopted = None
            else:
                result = self.run_once(job)
            job["pid"] = None
            if result == "done":
                job["status"], job["finished"], job["error"] = "done", now(), None
                self.save()
                log(f"{job['id']} 완료")
                return
            job["attempts"] += 1
            job["error"] = result
            if job["attempts"] >= self.max_attempts:
                job["status"], job["finished"] = "failed", now()
                self.save()
                log(f"{job['id']} {job['attempts']}회 실패({result}), 다음 작업으로")
                return
            job["status"] = "pending"
            self.save()
            log(f"{job['id']} 실패({result}), 재시도")
            time.sleep(10)

    def run(self) -> int:
        for job in self.queue["jobs"]:
            if job["status"] in ("done", "failed"):
                continue
            if is_done(job):
                job["status"], job["finished"] = "done", now()
                self.save()
                log(f"{job['id']} 결과 파일이 이미 있음, done")
                continue
            self.run_job(job)
        counts = self.counts()
        self.status(None, None, "모든 작업 종료")
        write_json(RUNTIME / "supervisor_done.json", {
            "time": now(), "counts": counts,
            "failed": [j["id"] for j in self.queue["jobs"] if j["status"] == "failed"]})
        log(f"모든 작업 종료: {counts}")
        return 0 if counts.get("failed", 0) == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", type=Path, default=RUNTIME / "queue.json")
    ap.add_argument("--python", default=sys.executable, help="작업을 실행할 파이썬(기본: 감독자와 같은 것)")
    ap.add_argument("--stall-minutes", type=float, default=20)
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--poll", type=float, default=10)
    args = ap.parse_args()

    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = FileLock(RUNTIME / "supervisor.lock")
    pid_file = RUNTIME / "supervisor.pid"
    try:
        with lock.acquire(timeout=0):
            pid_file.write_text(str(os.getpid()), encoding="utf-8")
            try:
                code = Supervisor(args.queue, args.python, args.stall_minutes, args.max_attempts, args.poll).run()
            finally:
                pid_file.unlink(missing_ok=True)
    except Timeout:
        print("다른 감독자가 이미 돌고 있음", flush=True)
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
