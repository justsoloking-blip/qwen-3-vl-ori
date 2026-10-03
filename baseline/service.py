#!/usr/bin/env python3
"""Manage this model service independently of the SSH connection."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

BASE = Path(__file__).resolve().parent
PID = BASE / "supervisor.pid"
CONFIG = json.loads((BASE / "config.json").read_text())
PYTHON = str(BASE / "venv/bin/python")


def active_pid():
    try:
        pid = int(PID.read_text())
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if str(BASE / "service.py").encode() in cmd and b"supervise" in cmd:
            return pid
    except (OSError, ValueError):
        pass
    return None


def supervise():
    stopping = False
    child = None

    def stop(_sig, _frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    failures = 0
    while not stopping:
        print(time.strftime("%FT%T"), "Starting model worker", flush=True)
        began = time.monotonic()
        child = subprocess.Popen(
            [PYTHON, str(BASE / "service.py"), "serve"],
            cwd=BASE, start_new_session=True,
        )
        stop_at = None
        while child.poll() is None:
            if stopping:
                if stop_at is None:
                    stop_at = time.monotonic()
                elif time.monotonic() - stop_at > 45:
                    os.killpg(child.pid, signal.SIGKILL)
            time.sleep(0.5)
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        print(time.strftime("%FT%T"), "Worker exited", child.returncode, flush=True)
        if stopping:
            break
        failures = failures + 1 if time.monotonic() - began < 600 else 1
        if failures >= 3:
            print("Stopped after three startup failures; inspect logs/server.log", flush=True)
            break
        for _ in range(20):
            if stopping:
                break
            time.sleep(1)


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "serve":
        os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["gpus"]
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "8"
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(BASE / "cache/torchinductor")
        os.environ["TRITON_CACHE_DIR"] = str(BASE / "cache/triton")
        import runpy
        sys.argv = ["sglang.launch_server", *CONFIG["arguments"],
                    "--api-key", (BASE / "api.key").read_text().strip()]
        runpy.run_module("sglang.launch_server", run_name="__main__")
        return
    if command == "supervise":
        supervise()
        return
    with (BASE / "control.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pid = active_pid()
        if command == "start":
            if pid:
                print(f"Already running: supervisor {pid}")
                return
            os.umask(0o077)
            with (BASE / "logs/server.log").open("ab", buffering=0) as log:
                proc = subprocess.Popen(
                    [PYTHON, str(BASE / "service.py"), "supervise"], cwd=BASE,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    start_new_session=True,
                )
            PID.write_text(str(proc.pid))
            print(f"Started supervisor {proc.pid}; model warmup is asynchronous")
        elif command == "stop":
            if pid:
                os.kill(pid, signal.SIGTERM)
                for _ in range(110):
                    if active_pid() is None:
                        break
                    time.sleep(0.5)
                if active_pid():
                    raise SystemExit("Supervisor has not stopped; inspect process before retrying")
            PID.unlink(missing_ok=True)
            print("Stopped")
        elif command == "status":
            print(json.dumps({"supervisor_pid": pid, "base_url": CONFIG["base_url"]}))
            try:
                req = urllib.request.Request(
                    "http://127.0.0.1:30000/v1/models",
                    headers={"Authorization": "Bearer " + (BASE / "api.key").read_text().strip()},
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    print(resp.read().decode())
            except Exception as exc:
                print(f"API not ready: {exc}")
                raise SystemExit(1)
        else:
            raise SystemExit("Usage: service.py start|stop|status")


if __name__ == "__main__":
    main()
