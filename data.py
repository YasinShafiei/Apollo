import os
import json
import math
import random

import torch
from torch.utils.data import IterableDataset, DataLoader

import pyarrow.parquet as pq
import tiktoken


class FineWebDataset(IterableDataset):
    def __init__(self, manifest_path, max_seq_len, tokenizer_name="cl100k_base", seed=42, shuffle=True):
        self.manifest_path = manifest_path
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.shuffle = shuffle

        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.eot = self.enc.eot_token

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.files = [json.loads(l)["path"] for l in f if l.strip]

    def _worker_files(self):
        info = torch.utils.data.get_worker_info()
        if info is None:
            return self.files

        per_worker = int(math.ceil(len(self.files) / info.num_workers))
        start = info.id * per_worker
        end = min(start + per_worker, len(self.files))
        return self.files[start:end]

    def _iter_texts_from_parquet(self, path):
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["text"])
            col = table.column("text")
            for t in col.to_pylist():
                if t:
                    yield t

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = 0 if info is None else info.id

        rng = random.Random(self.seed + wid * 1000)
        files = self._worker_files()
        if self.shuffle:
            rng.shuffle(files)

        buf = []
        for path in files:
            for text in self._iter_texts_from_parquet(path):
                ids = self.enc.encode_ordinary(text)
                if ids:
                    buf.extend(ids)
                    buf.append(self.eot)

                while len(buf) >= self.max_seq_len + 1:
                    chunk = buf[: self.max_seq_len + 1]
                    del buf[: self.max_seq_len]

                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    yield x, y


if __name__ == "__main__":
    dataset = FineWebDataset(
        manifest_path="data/fineweb_edu_train.jsonl",
        max_seq_len=1024,
        tokenizer_name="gpt2",
        seed=42,
        shuffle=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=4,
        pin_memory=True,
    )

    x, y = next(iter(dataset))
    print("single sample x:", x.shape, "y:", y.shape)

    xb, yb = next(iter(loader))
    print("one batch xb:", xb.shape, "yb:", yb.shape)