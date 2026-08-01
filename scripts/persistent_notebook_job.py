#!/usr/bin/env python3
"""Run notebook subprocesses outside the kernel with persistent logs and status."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _command_hash(command: list[str], cwd: Path, fingerprint: str | None = None) -> str:
    payload = json.dumps(
        {"command": command, "cwd": str(cwd.resolve()), "fingerprint": fingerprint},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _process_active(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        return state != "Z" and b"persistent_notebook_job.py" in command
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
        return False


def _tail(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return "(log unavailable)"
    with path.open(encoding="utf-8", errors="replace") as stream:
        return "".join(deque(stream, maxlen=lines))


def run_persistent_job(
    project_root: str | Path,
    name: str,
    command: list[str],
    required_outputs: list[str | Path] | None = None,
    report_seconds: int = 300,
    workdir: str | Path | None = None,
    fingerprint: str | None = None,
) -> dict:
    """Launch or reconnect to a detached job and wait with bounded output."""
    root = Path(project_root).resolve()
    state_dir = root / "outputs" / ".notebook-jobs" / name
    state_dir.mkdir(parents=True, exist_ok=True)
    spec_path = state_dir / "spec.json"
    status_path = state_dir / "status.json"
    pid_path = state_dir / "worker.pid"
    log_path = state_dir / "job.log"
    normalized_command = [str(item) for item in command]
    outputs = [str(Path(item).resolve()) for item in required_outputs or []]
    cwd = Path(workdir).resolve() if workdir is not None else root
    command_hash = _command_hash(normalized_command, cwd, fingerprint)
    spec = {
        "name": name,
        "command": normalized_command,
        "cwd": str(cwd),
        "required_outputs": outputs,
        "fingerprint": fingerprint,
        "command_hash": command_hash,
    }

    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    if (
        status.get("state") == "completed"
        and status.get("command_hash") == command_hash
        and all(Path(path).exists() for path in outputs)
    ):
        print(f"[{name}] trabajo ya completado y validado. Log: {log_path}")
        return status

    pid = int(pid_path.read_text().strip()) if pid_path.is_file() else -1
    process = None
    if _process_active(pid):
        existing = json.loads(spec_path.read_text(encoding="utf-8"))
        if existing.get("command_hash") != command_hash:
            raise RuntimeError(f"[{name}] hay otro comando activo con PID {pid}")
        print(f"[{name}] reconectado al PID {pid}. Log: {log_path}")
    else:
        if log_path.is_file() and (
            status.get("state") in {"completed", "failed"}
            or status.get("command_hash") != command_hash
        ):
            archived_log = state_dir / f"job-{int(time.time())}.log"
            os.replace(log_path, archived_log)
            print(f"[{name}] log anterior archivado en {archived_log}")
        _atomic_json(spec_path, spec)
        _atomic_json(
            status_path,
            {"state": "launching", "command_hash": command_hash, "updated_at": time.time()},
        )
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "worker", str(spec_path)],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        pid = process.pid
        pid_path.write_text(f"{pid}\n", encoding="utf-8")
        print(f"[{name}] iniciado como PID {pid}. Log: {log_path}")

    last_report = time.monotonic()
    while True:
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        if status.get("state") in {"completed", "failed"}:
            break
        if not _process_active(pid):
            raise RuntimeError(f"[{name}] worker desaparecio sin estado final. Log:\n{_tail(log_path)}")
        time.sleep(10)
        if time.monotonic() - last_report >= report_seconds:
            print(f"[{name}] PID {pid} activo. Log: {log_path}")
            last_report = time.monotonic()

    if process is not None:
        process.wait()
    if status["state"] != "completed":
        raise RuntimeError(f"[{name}] fallo con codigo {status.get('returncode')}. Log:\n{_tail(log_path)}")
    missing = [path for path in outputs if not Path(path).exists()]
    if missing:
        raise RuntimeError(f"[{name}] termino sin outputs requeridos: {missing}")
    print(f"[{name}] completado. Log: {log_path}")
    return status


def worker(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    state_dir = spec_path.parent
    log_path = state_dir / "job.log"
    status_path = state_dir / "status.json"
    status = {
        "state": "running",
        "command_hash": spec["command_hash"],
        "pid": os.getpid(),
        "started_at": time.time(),
    }
    _atomic_json(status_path, status)
    with log_path.open("ab", buffering=0) as log:
        result = subprocess.run(
            spec["command"],
            cwd=spec["cwd"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    status.update(
        {
            "state": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "finished_at": time.time(),
        }
    )
    _atomic_json(status_path, status)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["worker"])
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    return worker(args.spec)


if __name__ == "__main__":
    raise SystemExit(main())
