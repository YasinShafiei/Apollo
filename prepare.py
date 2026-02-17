import os
import math
import hashlib
from typing import Dict, List

# Check for filelock compatibility issue before importing huggingface libraries
def _check_dependencies():
    try:
        import filelock
        # Check if filelock supports the 'mode' parameter (added in filelock >= 3.4.1)
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

import numpy as np
import tiktoken
from tqdm import tqdm
from datasets import load_dataset

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-100BT"
VAL_RATIO = 0.0005
SEED = 2357
DTYPE = np.uint16
NUM_PROC = max(1, os.cpu_count() // 2)
BATCH_SIZE_MAP = 1000
WRITE_CHUNK_DOCS = 2048
STREAMING = False
OUT_DIR = "data"
HF_SPLIT = None

_enc = tiktoken.get_encoding("gpt2")
_eot = _enc.eot_token

def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, List]:
    texts = batch["text"]
    ids = []
    lens = []
    for t in texts:
        tok = _enc.encode_ordinary(t)
        tok.append(_eot)
        ids.append(tok)
        lens.append(len(tok))
    return {"ids": ids, "len": lens}

def _stable_hash_u64(s: str) -> int:
    h = hashlib.blake2b(s.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)

def prepare_cached():
    split = HF_SPLIT or "train"
    ds = load_dataset(
    DATASET_NAME,
    DATASET_CONFIG,             
    split=split,
    streaming=STREAMING,
    )
    split_ds = ds.train_test_split(test_size=VAL_RATIO, seed=SEED, shuffle=True)
    split_ds["val"] = split_ds.pop("test")
    tok = split_ds.map(
        tokenize_batch,
        batched=True,
        batch_size=BATCH_SIZE_MAP,
        remove_columns=["text"],
        num_proc=NUM_PROC,
        desc="tokenizing",
    )

    os.makedirs(OUT_DIR, exist_ok=True)

    for split_name, dset in tok.items():
        arr_len = int(np.sum(dset["len"], dtype=np.uint64))
        out_path = os.path.join(OUT_DIR, f"{split_name}.bin")
        print(f"\nwriting {out_path} ({arr_len:,} tokens) ...")

        arr = np.memmap(out_path, dtype=DTYPE, mode="w+", shape=(arr_len,))
        idx = 0
        dset_np = dset.with_format("numpy")
        n = len(dset_np)
        n_chunks = math.ceil(n / WRITE_CHUNK_DOCS)

        for c in tqdm(range(n_chunks), desc=f"writing {split_name}"):
            start = c * WRITE_CHUNK_DOCS
            end = min(n, (c + 1) * WRITE_CHUNK_DOCS)
            batch = dset_np[start:end]
            flat = np.concatenate(batch["ids"]).astype(DTYPE, copy=False)
            arr[idx: idx + flat.size] = flat
            idx += flat.size

        arr.flush()
        assert idx == arr_len, f"wrote {idx} tokens, expected {arr_len}"

    print("\nDone. train.bin and val.bin are ready.")

def prepare_streaming():
    split = HF_SPLIT or "train"
    it = load_dataset(DATASET_NAME, split=split, streaming=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    train_path = os.path.join(OUT_DIR, "train.bin")
    val_path = os.path.join(OUT_DIR, "val.bin")
    ft = open(train_path, "wb")
    fv = open(val_path, "wb")
    buf_t = []
    buf_v = []
    buf_t_n = 0
    buf_v_n = 0
    FLUSH_TOKENS = 8_000_000

    train_tokens = 0
    val_tokens = 0
    docs = 0

    print("\nstreaming + tokenizing + writing ...")
    for ex in tqdm(it):
        text = ex["text"]
        docs += 1

        tok = _enc.encode_ordinary(text)
        tok.append(_eot)
        h = _stable_hash_u64(f"{SEED}:{text[:256]}")
        in_val = (h % 1_000_000) < int(VAL_RATIO * 1_000_000)

        if in_val:
            buf_v.append(tok)
            buf_v_n += len(tok)
        else:
            buf_t.append(tok)
            buf_t_n += len(tok)

        if buf_t_n >= FLUSH_TOKENS:
            flat = np.asarray([x for xs in buf_t for x in xs], dtype=DTYPE)
            flat.tofile(ft)
            train_tokens += flat.size
            buf_t.clear()
            buf_t_n = 0

        if buf_v_n >= FLUSH_TOKENS:
            flat = np.asarray([x for xs in buf_v for x in xs], dtype=DTYPE)
            flat.tofile(fv)
            val_tokens += flat.size
            buf_v.clear()
            buf_v_n = 0

    if buf_t:
        flat = np.asarray([x for xs in buf_t for x in xs], dtype=DTYPE)
        flat.tofile(ft)
        train_tokens += flat.size
    if buf_v:
        flat = np.asarray([x for xs in buf_v for x in xs], dtype=DTYPE)
        flat.tofile(fv)
        val_tokens += flat.size

    ft.close()
    fv.close()

    print(f"\nDone. train.bin ({train_tokens:,} tokens), val.bin ({val_tokens:,} tokens), docs seen: {docs:,}")
    print("Note: streaming mode avoids the huge HF cache, but can be slower than cached mode.")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    if STREAMING:
        prepare_streaming()
    else:
        prepare_cached()