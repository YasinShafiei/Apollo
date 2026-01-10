import json
import numpy as np

import torch
from torch.utils.data import Dataset

import tiktoken

# prompt template for instruction following
PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Response:
{response}"""

PROMPT_TEMPLATE_WITH_INPUT = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}"""


class SFTDataset(Dataset):
    def __init__(self, data_path, max_seq_len, mask_prompt=True):
        self.max_seq_len = max_seq_len
        self.mask_prompt = mask_prompt
        
        # load tokenizer
        self.enc = tiktoken.get_encoding("gpt2")
        self.pad_token = self.enc.eot_token
        
        # load and process data
        self.examples = self._load_data(data_path)
    
    def _load_data(self, data_path):
        with open(data_path, "r") as f:
            raw_data = json.load(f)
        
        examples = []
        for item in raw_data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")
            
            # skip empty examples
            if not instruction or not output:
                continue
            
            # format prompt based on whether input exists
            if input_text:
                prompt = PROMPT_TEMPLATE_WITH_INPUT.format(
                    instruction=instruction,
                    input=input_text,
                    response=""
                )
                full_text = PROMPT_TEMPLATE_WITH_INPUT.format(
                    instruction=instruction,
                    input=input_text,
                    response=output
                )
            else:
                prompt = PROMPT_TEMPLATE.format(
                    instruction=instruction,
                    response=""
                )
                full_text = PROMPT_TEMPLATE.format(
                    instruction=instruction,
                    response=output
                )
            
            # tokenize
            prompt_ids = self.enc.encode(prompt)
            full_ids = self.enc.encode(full_text)
            full_ids.append(self.enc.eot_token)
            
            # skip if too long
            if len(full_ids) > self.max_seq_len:
                continue
            
            examples.append({
                "input_ids": full_ids,
                "prompt_len": len(prompt_ids)
            })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        input_ids = example["input_ids"]
        prompt_len = example["prompt_len"]
        
        # pad to max_seq_len
        pad_len = self.max_seq_len - len(input_ids) + 1
        input_ids = input_ids + [self.pad_token] * pad_len
        
        # create input and target
        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(input_ids[1:], dtype=torch.long)
        
        # create loss mask (only compute loss on response tokens)
        if self.mask_prompt:
            loss_mask = torch.zeros(self.max_seq_len, dtype=torch.float)
            # mask prompt tokens and padding
            response_start = prompt_len - 1  # -1 because of shift
            response_end = len(example["input_ids"]) - 1
            loss_mask[response_start:response_end] = 1.0
        else:
            # compute loss on all non-padding tokens
            loss_mask = torch.ones(self.max_seq_len, dtype=torch.float)
            loss_mask[len(example["input_ids"])-1:] = 0.0
        
        return x, y, loss_mask


class SFTSampler:
    def __init__(self, dataset_size, batch_size, rank, world_size, shuffle=True, seed=42):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # calculate samples per rank
        self.samples_per_rank = dataset_size // world_size
        
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def __iter__(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        
        # create shuffled indices
        if self.shuffle:
            indices = torch.randperm(self.dataset_size, generator=rng).tolist()
        else:
            indices = list(range(self.dataset_size))
        
        # split indices across ranks
        start_idx = self.rank * self.samples_per_rank
        end_idx = start_idx + self.samples_per_rank
        rank_indices = indices[start_idx:end_idx]
        
        for idx in rank_indices:
            yield idx
    
    def __len__(self):
        return self.samples_per_rank


def download_alpaca():
    """Download Alpaca dataset from HuggingFace."""
    from datasets import load_dataset
    
    print("downloading alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca")
    
    # save to json
    data = []
    for item in dataset["train"]:
        data.append({
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["output"]
        })
    
    with open("alpaca.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"saved {len(data)} examples to alpaca.json")


if __name__ == "__main__":
    download_alpaca()