import os
import json
import hashlib
import time
from typing import List, Dict

# Check for filelock compatibility issue before importing huggingface libraries
def _check_dependencies():
    try:
        import filelock
        import inspect
        sig = inspect.signature(filelock.FileLock.__init__)
        if 'mode' not in sig.parameters:
            print("ERROR: Your 'filelock' package is too old and incompatible with huggingface_hub.")
            print("Please upgrade it by running:")
            print("  pip install --upgrade filelock")
            print("Or if using system packages:")
            print("  pip install --user --upgrade filelock")
            exit(1)
    except ImportError:
        pass  # filelock not installed, let later imports handle it

_check_dependencies()

from huggingface_hub import snapshot_download, HfApi
import pyarrow.parquet as pq

DATASET_REPO = "HuggingFaceFW/fineweb-edu"
DATASET_SUBDIR = "sample/100BT"

OUT_DIR = "data"
PARQUET_DIR = os.path.join(OUT_DIR, "fineweb_edu", "parquet")

VAL_RATIO = 0.0005
SEED = 2357

# Optional: limit how many parquet shards you download (useful for testing).
MAX_FILES = None

def _stable_hash_u64(s: str) -> int:
    h = hashlib.blake2b(s.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)

def _list_parquet_files(repo_id: str, subdir: str) -> List[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    pref = subdir.rstrip("/") + "/"
    out = [f for f in files if f.startswith(pref) and f.endswith(".parquet")]
    out.sort()
    return out

def _split_files(files: List[str]) -> Dict[str, List[str]]:
    train, val = [], []
    thresh = int(VAL_RATIO * 1_000_000)
    for f in files:
        h = _stable_hash_u64(f"{SEED}:{f}")
        in_val = (h % 1_000_000) < thresh
        (val if in_val else train).append(f)
    return {"train": train, "val": val}

def _write_manifest_jsonl(manifest_path: str, local_paths: List[str], split_name: str):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    n_files = 0
    total_rows = 0
    with open(manifest_path, "w", encoding="utf-8") as w:
        for p in local_paths:
            pf = pq.ParquetFile(p)
            md = pf.metadata
            rows = int(md.num_rows) if md is not None else 0
            rgs = int(md.num_row_groups) if md is not None else 0
            w.write(json.dumps({"path": p, "split": split_name, "rows": rows, "row_groups": rgs}) + "\n")
            n_files += 1
            total_rows += rows
    print(f"{split_name}: wrote manifest {manifest_path} ({n_files} files, {total_rows:,} rows)")

def prepare():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PARQUET_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print(f"listing parquet shards in {DATASET_REPO}/{DATASET_SUBDIR} ...")
    repo_files = _list_parquet_files(DATASET_REPO, DATASET_SUBDIR)
    if not repo_files:
        raise RuntimeError(f"no parquet files found under {DATASET_SUBDIR}")

    if MAX_FILES is not None:
        repo_files = repo_files[:MAX_FILES]
        print(f"MAX_FILES is set -> using only {len(repo_files)} shards")

    splits = _split_files(repo_files)
    print(f"split: train={len(splits['train'])} files, val={len(splits['val'])} files (VAL_RATIO={VAL_RATIO})")

    allow = splits["train"] + splits["val"]
    print(f"downloading {len(allow)} parquet files into {PARQUET_DIR} ...")

    # This writes actual files under data/, making the dataset self-contained.
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        allow_patterns=allow,
        local_dir=PARQUET_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    # snapshot_download preserves the repo file tree under local_dir
    train_local = [os.path.join(PARQUET_DIR, f) for f in splits["train"]]
    val_local = [os.path.join(PARQUET_DIR, f) for f in splits["val"]]

    train_manifest = os.path.join(OUT_DIR, "fineweb_edu_train.jsonl")
    val_manifest = os.path.join(OUT_DIR, "fineweb_edu_val.jsonl")

    print("building manifests (this reads parquet metadata only) ...")
    _write_manifest_jsonl(train_manifest, train_local, "train")
    _write_manifest_jsonl(val_manifest, val_local, "val")

    print("\nDone.")
    print(f"parquets:       {PARQUET_DIR}")
    print(f"train manifest: {train_manifest}")
    print(f"val manifest:   {val_manifest}")

if __name__ == "__main__":
    start_time = time.time()
    prepare()
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\n{minutes}m, {seconds}s")
