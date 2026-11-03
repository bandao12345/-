"""Optional local llama.cpp lifecycle; no model download at application startup."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parent


def load_settings():
    path = ROOT / "local_settings.json"
    if not path.is_file():
        return
    if path.stat().st_size > 16384:
        raise ValueError("local_settings.json is too large")
    settings = json.loads(path.read_text("utf-8-sig"))
    for key in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "INKLAB_DISABLE_LOCAL_LLM"):
        value = settings.get(key)
        if isinstance(value, str) and value and key not in os.environ:
            os.environ[key] = value


def _ready():
    try:
        with build_opener(ProxyHandler({})).open("http://127.0.0.1:8087/v1/models", timeout=1) as response:
            data = json.load(response)
        return any(model.get("id") == "inklab-local" for model in data.get("data", []))
    except Exception:
        return False


def start_optional_model():
    """Return an owned process, or None when reusing/disabled/unavailable.

    Only the verified, fixed local model runtime can be launched. Neither HTTP
    requests nor imported documents supply executable names or process args.
    """
    load_settings()
    if os.environ.get("LLM_BASE_URL") or os.environ.get("INKLAB_DISABLE_LOCAL_LLM") == "1":
        return None
    model = ROOT / "models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    executable = ROOT / ".runtime/llama/bin/llama-server.exe"
    if not executable.is_file():
        executable = ROOT.parents[1] / ".tools/llama/bin/llama-server.exe"
    if not model.is_file() or not executable.is_file():
        return None
    process = None
    if not _ready():
        log_dir = Path(os.environ.get("INKLAB_DATA_DIR", ROOT / "instance")) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        print("Starting optional local Qwen model (CPU, no API key)...", flush=True)
        with (log_dir / "local-model.log").open("ab") as log:
            process = subprocess.Popen([
                str(executable), "-m", str(model), "--host", "127.0.0.1", "--port", "8087",
                "--alias", "inklab-local", "-c", "4096", "-t", "4", "-ngl", "0",
                "--parallel", "1", "--no-webui",
            ], cwd=str(executable.parent), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(50):
            if _ready():
                break
            if process.poll() is not None:
                print("Local model could not start; document retrieval remains available.", flush=True)
                return None
            time.sleep(.4)
        else:
            process.terminate()
            process.wait(timeout=10)
            print("Local model startup timed out; document retrieval remains available.", flush=True)
            return None
    os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8087/v1"
    os.environ["LLM_MODEL"] = "inklab-local"
    return process


def stop_owned_model(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

