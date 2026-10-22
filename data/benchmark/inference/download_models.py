#!/usr/bin/env python3
"""Download a QA model manifest to its explicit local paths, with resumable state."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    state_path = args.manifest.parent / "downloads.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    lock = threading.Lock()

    def update(alias, **fields):
        with lock:
            state.setdefault(alias, {}).update(fields, updated_at=datetime.now(timezone.utc).isoformat())
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(state_path)

    def download(model):
        alias, repo, path = model["id"], model["repo"], Path(model["path"])
        if state.get(alias, {}).get("status") == "complete" and (path / "config.json").exists():
            return
        for attempt in range(1, 4):
            try:
                update(alias, status="downloading", repo=repo, path=str(path), attempt=attempt)
                info = HfApi().model_info(repo)
                snapshot_download(repo, revision=info.sha, local_dir=path, max_workers=4,
                                  allow_patterns=["*.json", "*.safetensors", "*.py", "*.model",
                                                  "*.txt", "*.tiktoken", "*.jinja", "*.jinja2",
                                                  "*.yaml", "*.yml", "README.md", "LICENSE*"])
                weights = [item.rfilename for item in info.siblings if item.rfilename.endswith(".safetensors")]
                missing = [name for name in weights if not (path / name).is_file()]
                if not weights or missing:
                    raise ValueError(f"Missing checkpoint weights: {missing or 'no safetensors published'}")
                total = sum((path / name).stat().st_size for name in weights)
                update(alias, status="complete", revision=info.sha, weight_files=len(weights), weight_bytes=total)
                print(f"READY {alias}: {len(weights)} files, {total / 2**30:.2f} GiB", flush=True)
                return
            except Exception as error:
                message = str(error)
                if attempt == 3 or any(code in message for code in ("401", "403", "gated repo")):
                    update(alias, status="blocked", error=message)
                    print(f"BLOCKED {alias}: {message}", flush=True)
                    return
                print(f"RETRY {alias}: {message}", flush=True)
                time.sleep(5)

    models = json.loads(args.manifest.read_text())["models"]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed(pool.submit(download, model) for model in models):
            future.result()


if __name__ == "__main__":
    main()
