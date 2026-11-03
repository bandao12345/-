"""Install pinned, hash-checked optional Qwen/llama.cpp assets on Windows x64.

Usage: python tools/download_local_model.py [--check]
No API keys are needed. Startup never invokes this downloader automatically.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matches(path, metadata):
    return (path.is_file() and path.stat().st_size == metadata["bytes"]
            and sha256(path) == metadata["sha256"])


def fetch(metadata, destination):
    if matches(destination, metadata):
        print("Verified existing:", destination.name, flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(metadata["url"], headers={"User-Agent": "INKLAB-setup/3.0"})
    try:
        written = 0
        with urllib.request.urlopen(request, timeout=90) as response, pending.open("wb") as stream:
            while block := response.read(1024 * 1024):
                written += len(block)
                if written > metadata["bytes"]:
                    raise ValueError("Downloaded data exceeds the pinned file size")
                stream.write(block)
        if not matches(pending, metadata):
            raise ValueError("Downloaded file size or SHA-256 differs from manifest.json")
        pending.replace(destination)
        print("Downloaded and verified:", destination.name, flush=True)
    finally:
        pending.unlink(missing_ok=True)


def extract_runtime(archive, target):
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        if sum(item.file_size for item in bundle.infolist()) > 512 * 1024 * 1024:
            raise ValueError("Runtime archive is unexpectedly large")
        for item in bundle.infolist():
            if not (target / item.filename).resolve().is_relative_to(target):
                raise ValueError("Unsafe ZIP path")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("ZIP links are not supported")
        bundle.extractall(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify installed assets; do not download")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "models/llm/manifest.json").read_text("utf-8"))
    model = ROOT / "models/llm" / manifest["model"]["filename"]
    local_runtime = ROOT / ".runtime/llama"
    shared_runtime = ROOT.parents[1] / ".tools/llama"
    if args.check:
        runtime = local_runtime if (local_runtime / "bin/llama-server.exe").is_file() else shared_runtime
        if not matches(model, manifest["model"]):
            raise SystemExit("Local model is missing or its hash differs. Run this script without --check.")
        executable = runtime / "bin/llama-server.exe"
        if not executable.is_file() or sha256(executable) != manifest["runtime"]["server_sha256"]:
            raise SystemExit("Local llama.cpp runtime is missing or its hash differs.")
        print("Qwen model and llama.cpp server hashes verified. No download performed.")
        return
    if platform.system() != "Windows" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise SystemExit("This pinned runtime supports Windows x64. Use a separate OpenAI-compatible LLM on other systems.")
    print("Optional download: 1.04 GiB Qwen weights + 23 MiB CPU runtime.", flush=True)
    fetch(manifest["model"], model)
    archive = local_runtime / manifest["runtime"]["filename"]
    fetch(manifest["runtime"], archive)
    extract_runtime(archive, local_runtime / "bin")
    if sha256(local_runtime / "bin/llama-server.exe") != manifest["runtime"]["server_sha256"]:
        raise SystemExit("Extracted server hash differs from manifest.json")
    print("Ready. Restart start.bat to enable the optional local model.")


if __name__ == "__main__":
    main()
