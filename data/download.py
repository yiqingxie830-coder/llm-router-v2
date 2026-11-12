"""download.py — download the Arena human preference data (stage 1, step 0)

Decisions
---------
1. Source: lmarena-ai/arena-human-preference-140k (2025-04 to 2025-07, 135,634 text-only battles).
   - The older lmsys/arena-human-preference-100k (2024 models) and 55k sets were dropped;
     on 2026-11-07 we decided to use only the newest 140k set.
   - The lmsys/ organization was renamed lmarena-ai/; old URLs redirect or return 401.
2. Only the 7 parquet shards under data/ are fetched, about 1.6 GB in total.
3. Downloads are treated as untrusted files: they go into their own directory (never next to
   the scripts), and each one is checked against the LFS SHA-256 that Hugging Face publishes,
   so we know it is the official file, not truncated or altered. A mismatch deletes it and fails.
4. Standard library only (urllib), no huggingface_hub: fewer dependencies, easier to reproduce.
5. Network: a sandboxed environment must allow huggingface.co and us.aws.cdn.hf.co (the file bytes come from the latter).

Usage: python -I download.py <output dir>
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = "lmarena-ai/arena-human-preference-140k"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/data"
FILE = f"https://huggingface.co/datasets/{REPO}/resolve/main/{{path}}"


def list_files():
    """Get the shard list and each file's official SHA-256 (LFS oid) from the HF API."""
    with urllib.request.urlopen(API, timeout=60) as r:
        return [(x["path"], x["lfs"]["oid"]) for x in json.load(r) if x["path"].endswith(".parquet")]


def download(path: str, sha: str, out_dir: Path) -> Path:
    dest = out_dir / Path(path).name
    h = hashlib.sha256()
    with urllib.request.urlopen(FILE.format(path=path), timeout=600) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
    if h.hexdigest() != sha:
        dest.unlink()
        raise ValueError(f"sha256 mismatch, deleted {dest}")
    return dest


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_files()
    for path, sha in files:
        print(f"OK {download(path, sha, out_dir)} (sha256 verified)")
    (out_dir / "SOURCE.txt").write_text(
        f"repo={REPO}\n" + "".join(f"{p} sha256={s}\n" for p, s in files))


if __name__ == "__main__":
    main()
