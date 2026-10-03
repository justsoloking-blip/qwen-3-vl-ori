#!/usr/bin/env python3
"""Prepare the student baseline runtime. Never starts a GPU service."""
import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def check_model(directory):
    required = ["config.json", "tokenizer.json", "tokenizer_config.json",
                "preprocessor_config.json", "video_preprocessor_config.json",
                "model.safetensors.index.json"]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise SystemExit(f"Incomplete model snapshot: missing {', '.join(missing)}")
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    shards = set(index.get("weight_map", {}).values())
    if not shards:
        raise SystemExit("Model index has no weight shards")
    for name in shards:
        file = (directory / name).resolve()
        if not file.is_relative_to(directory) or not file.is_file() or not file.stat().st_size:
            raise SystemExit(f"Missing, empty or invalid weight shard: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    parser.add_argument("--gpus", help="Visible GPU indices; does not change tp-size")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--runtime-root", type=Path, default=ROOT / "runtime")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    target = (args.runtime_root / "baseline").resolve()
    source = ROOT / "baseline"
    if target.exists():
        raise SystemExit(f"{target} already exists. Choose another --runtime-root to preserve its state.")
    model = args.model_path.resolve()
    check_model(model)
    venv = args.venv.resolve()
    python = venv / "bin/python"
    if not python.is_file():
        raise SystemExit("Create the clean Linux Python 3.12 virtual environment first.")
    metadata_code = (
        'import importlib.metadata as m,importlib.util,json,sys;'
        'print(json.dumps({"version":m.version("sglang"),'
        '"python":list(sys.version_info[:2]),'
        '"path":str(next(iter(importlib.util.find_spec("sglang").submodule_search_locations)))}))'
    )
    info = json.loads(subprocess.check_output([str(python), "-c", metadata_code], text=True))
    if info["version"] != "0.5.6.post2" or info["python"] != [3, 12]:
        raise SystemExit("This baseline requires Python 3.12 and sglang==0.5.6.post2")
    config = json.loads((source / "config.json").read_text())
    arguments = config["arguments"]
    arguments[arguments.index("--model-path") + 1] = str(model)
    arguments[arguments.index("--port") + 1] = str(args.port)
    config["base_url"] = f"http://127.0.0.1:{args.port}/v1"
    if args.gpus:
        config["gpus"] = args.gpus
    old_umask = os.umask(0o077)
    try:
        target.mkdir(parents=True)
        for name in ("logs", "results", "cache"):
            (target / name).mkdir()
        (target / "venv").symlink_to(venv, target_is_directory=True)
        shutil.copytree(info["path"], target / "overlay/sglang",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for item in (source / "src").rglob("*.py"):
            out = target / "overlay" / item.relative_to(source / "src")
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, out)
        service = (source / "service.py").read_text()
        # Port and import-path adaptation only; no model/parallel/kernel fixes.
        service = service.replace('"http://127.0.0.1:30000/v1/models"',
                                  'CONFIG["base_url"] + "/models"')
        service = service.replace(
            'BASE = Path(__file__).resolve().parent',
            'BASE = Path(__file__).resolve().parent\n'
            'sys.path.insert(0, str(BASE / "overlay"))\n'
            'os.environ["PYTHONPATH"] = str(BASE / "overlay")')
        (target / "service.py").write_text(service)
        (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (target / "runtime-origin.json").write_text(json.dumps(info, indent=2) + "\n")
        (target / "api.key").write_text(secrets.token_urlsafe(32) + "\n")
        (target / "api.key").chmod(0o600)
    finally:
        os.umask(old_umask)
    print(f"Prepared {target}; API key stored with mode 0600, never printed.")
    print(f"Start: {python} {target}/service.py start")


if __name__ == "__main__":
    main()

