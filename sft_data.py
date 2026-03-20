import os
import json
import random

import torch
from torch.utils.data import IterableDataset

import tiktoken


# ChatML special token IDs for cl100k_base
IM_START = 100264  # <|im_start|>
IM_END   = 100265  # <|im_end|>
CHOICES  = ["A", "B", "C", "D"]


def prepare_sft_data(data_dir="data/sft"):
    """Download SmolTalk, MMLU, GSM8K from HuggingFace and format as conversation JSONL."""
    from datasets import load_dataset

    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "sft_train.jsonl")
    val_path   = os.path.join(data_dir, "sft_val.jsonl")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print("SFT data already prepared, skipping download")
        return

    train_convs = []
    val_convs   = []

    print("Downloading SmolTalk...")
    ds = load_dataset("HuggingFaceTB/smol-smoltalk")
    for item in ds["train"]:
        msgs = [{"role": m["role"], "content": m["content"]}
                for m in item["messages"] if m["role"] in ("user", "assistant")]
        if len(msgs) >= 2:
            train_convs.append({"messages": msgs})
    for item in ds["test"]:
        msgs = [{"role": m["role"], "content": m["content"]}
                for m in item["messages"] if m["role"] in ("user", "assistant")]
        if len(msgs) >= 2:
            val_convs.append({"messages": msgs})
    print(f"  SmolTalk: {len(train_convs)} train, {len(val_convs)} val")

    print("Downloading MMLU...")
    ds = load_dataset("cais/mmlu", "all")

    mmlu_train = []
    for item in ds["auxiliary_train"]:
        q = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]
        user_msg = q + "\n"
        for i, c in enumerate(choices):
            user_msg += f"{CHOICES[i]}) {c}\n"
        user_msg += "\nAnswer with the letter only."
        mmlu_train.append({"messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": CHOICES[answer_idx]},
        ]})
    train_convs.extend(mmlu_train * 3)  # 3 epochs

    for item in ds["validation"]:
        q = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]
        user_msg = q + "\n"
        for i, c in enumerate(choices):
            user_msg += f"{CHOICES[i]}) {c}\n"
        user_msg += "\nAnswer with the letter only."
        val_convs.append({"messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": CHOICES[answer_idx]},
        ]})
    print(f"  MMLU: {len(mmlu_train)}x3 train, added to val")

    print("Downloading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main")

    gsm_train = []
    for item in ds["train"]:
        gsm_train.append({"messages": [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["answer"]},
        ]})
    train_convs.extend(gsm_train * 4)  # 4 epochs

    for item in ds["test"]:
        val_convs.append({"messages": [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["answer"]},
        ]})
    print(f"  GSM8K: {len(gsm_train)}x4 train, added to val")


    random.seed(42)
    random.shuffle(train_convs)

    with open(train_path, "w") as f:
        for conv in train_convs:
            f.write(json.dumps(conv) + "\n")
    with open(val_path, "w") as f:
        for conv in val_convs:
            f.write(json.dumps(conv) + "\n")

    print(f"Saved {len(train_convs):,} train, {len(val_convs):,} val conversations")


class SFTMixtureDataset(IterableDataset):
    """
    Streaming packed SFT dataset with loss masking.
    Tokenizes conversations in ChatML format, packs them into fixed-length
    sequences, and yields (input, target, loss_mask) tuples.
    Matches FineWebDataset's streaming speed.
    """

    def __init__(self, data_path, max_seq_len, rank=0, world_size=1, seed=42, shuffle=True):
        self.max_seq_len = max_seq_len
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle

        self.enc = tiktoken.get_encoding("cl100k_base")
        self.eot = self.enc.eot_token

        # load conversations and shard across DDP ranks
        self.conversations = []
        with open(data_path, "r") as f:
            for i, line in enumerate(f):
                if line.strip() and i % world_size == rank:
                    self.conversations.append(json.loads(line))

    def _tokenize_conversation(self, conv):
        """Tokenize a conversation into ChatML format, returning (tokens, mask)."""
        tokens = []
        mask = []

        for msg in conv["messages"]:
            role = msg["role"]
            content = msg["content"]

            # <|im_start|>role\ncontent<|im_end|>
            role_tokens = self.enc.encode(role + "\n")
            content_tokens = self.enc.encode(content)

            msg_tokens = [IM_START] + role_tokens + content_tokens + [IM_END]

            if role == "assistant":
                # train on assistant content + im_end token
                msg_mask = [0] * (1 + len(role_tokens)) + [1] * (len(content_tokens) + 1)
            else:
                msg_mask = [0] * len(msg_tokens)

            tokens.extend(msg_tokens)
            mask.extend(msg_mask)

        # end-of-conversation separator
        tokens.append(self.eot)
        mask.append(0)

        return tokens, mask

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        convs = self.conversations

        if info is not None:
            convs = convs[info.id::info.num_workers]

        wid = 0 if info is None else info.id
        rng = random.Random(self.seed + self.rank * 1000 + wid * 100)
        if self.shuffle:
            convs = list(convs)
            rng.shuffle(convs)

        token_buf = []
        mask_buf = []

        for conv in convs:
            tokens, loss_mask = self._tokenize_conversation(conv)

            # skip extremely long conversations (> 2x context)
            if len(tokens) > self.max_seq_len * 2:
                continue

            token_buf.extend(tokens)
            mask_buf.extend(loss_mask)

            while len(token_buf) >= self.max_seq_len + 1:
                chunk_t = token_buf[: self.max_seq_len + 1]
                chunk_m = mask_buf[: self.max_seq_len + 1]
                del token_buf[: self.max_seq_len]
                del mask_buf[: self.max_seq_len]

                x = torch.tensor(chunk_t[:-1], dtype=torch.long)
                y = torch.tensor(chunk_t[1:],  dtype=torch.long)
                m = torch.tensor(chunk_m[1:],  dtype=torch.float)
                yield x, y, m


if __name__ == "__main__":
    prepare_sft_data()
