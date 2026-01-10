import numpy as np

import torch
from torch.utils.data import Dataset

class DatasetSampler:
    def __init__(self, dataset_size, batch_size, rank, world_size, shuffle=True, seed=42, samples_per_epoch=500_000):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        self.samples_per_epoch = samples_per_epoch
        
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def __iter__(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch + self.rank * 1000)
        
        for _ in range(self.samples_per_epoch):
            if self.shuffle:
                idx = torch.randint(0, self.dataset_size, (1,), generator=rng).item()
            else:
                idx = (_ + self.rank * self.samples_per_epoch) % self.dataset_size
            yield idx
    
    def __len__(self):
        return self.samples_per_epoch

class GPTDataset(Dataset):
    def __init__(self, filename, max_seq_len):
        self.filename = filename
        self.max_seq_len = max_seq_len
        self.data = None 

    def _init_db(self):
        # Initialize the memmap only once per process
        self.data = np.memmap(self.filename, dtype=np.uint16, mode='r')

    def __len__(self):
        if self.data is None:
            self._init_db()
        return len(self.data) - self.max_seq_len 
    
    def __getitem__(self, idx):
        if self.data is None:
            self._init_db()
    
        # Use int64 directly to avoid casting issues in the model
        buffer = self.data[idx:idx + self.max_seq_len + 1].astype(np.int64)
        x = torch.from_numpy(buffer[:-1])
        y = torch.from_numpy(buffer[1:])
        return x, y
