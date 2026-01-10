import os
import sys
import math
import time

import torch 
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from model import Model
from data import GPTDataset, DatasetSampler

VOCAB_SIZE = 50257
EMBED_DIM = 1024  
N_LAYERS = 20
N_HEADS = 16
N_KV_HEADS = 8
HIDDEN_DIM = 2048
MAX_SEQ_LEN = 1024

MICRO_BATCH_SIZE = 16
ACCUM_STEPS = 4

LR = 3e-4
MIN_LR = 3e-5

TOKEN_BUDGET = 10_000_000_000

WARMUP_STEPS = 1200

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

def is_main_process():
    return dist.get_rank() == 0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def num_steps(micro_batch_size, max_seq_len, accum_steps, world_size):
    effective_batch_size = micro_batch_size * accum_steps * world_size
    total_steps = TOKEN_BUDGET // (effective_batch_size * max_seq_len)
    return total_steps

def validate(model, val_loader, num_steps=20):
    model.eval()
    val_loss = 0.0
    val_iter = iter(val_loader)
    with torch.no_grad():
        for _ in range(num_steps):
            try:
                input, target = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                input, target = next(val_iter)
            input = input.to(model.device)
            target = target.to(model.device)
            out, loss = model(input, target)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / num_steps
    return avg_val_loss


def train(model, optimizer, train_loader, val_loader, local_rank, world_size):
    model.train()
    total_steps = num_steps(MICRO_BATCH_SIZE, MAX_SEQ_LEN, ACCUM_STEPS, world_size)
    effective_batch_size = MICRO_BATCH_SIZE * ACCUM_STEPS * world_size
    tokens_per_step = effective_batch_size * MAX_SEQ_LEN
    
    if is_main_process():
        print(f"training for {total_steps:,} steps")
        print(f"effective batch size: {effective_batch_size}")
        print(f"tokens per step: {tokens_per_step:,}")
    
    step = 0
    last_log_time = time.time()
    last_log_step = 0
    train_iter = iter(train_loader)
    
    if is_main_process():
        print("starting training loop", flush=True)
    
    while step < total_steps:
        optimizer.zero_grad()
        accum_loss = 0.0
        
        # gradient accumulation loop
        for micro_step in range(ACCUM_STEPS):
            try:
                input, target = next(train_iter)
            except StopIteration:
                train_loader.sampler.set_epoch(step)
                train_iter = iter(train_loader)
                input, target = next(train_iter)
            
            input = input.to(local_rank)
            target = target.to(local_rank)
            
            # disable gradient sync for all but the last micro step
            if micro_step < ACCUM_STEPS - 1:
                with model.no_sync():
                    out, loss = model(input, target)
                    scaled_loss = loss / ACCUM_STEPS
                    scaled_loss.backward()
            else:
                out, loss = model(input, target)
                scaled_loss = loss / ACCUM_STEPS
                scaled_loss.backward()
            
            accum_loss += loss.item()
        
        # average loss over accumulation steps
        avg_loss = accum_loss / ACCUM_STEPS
        
        step += 1
        
        # learning rate schedule
        if step < WARMUP_STEPS:
            coeff = step / WARMUP_STEPS
            for param_group in optimizer.param_groups:
                param_group['lr'] = coeff * LR
        else:
            coeff = 0.5 * (1.0 + math.cos(math.pi * ((step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS))))
            for param_group in optimizer.param_groups:
                param_group['lr'] = MIN_LR + coeff * (LR - MIN_LR)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if step == 1 and is_main_process():
            print(f"first step completed, loss: {avg_loss:.4f}", flush=True)

        if step % 100 == 0 and is_main_process():
            val_loss = validate(model, val_loader, num_steps=20)
            current_time = time.time()
            elapsed = current_time - last_log_time
            steps_since_last_log = step - last_log_step if step > 0 else 1
            tokens_per_sec = (steps_since_last_log * tokens_per_step) / elapsed if elapsed > 0 else 0
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step {step}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}, Tokens/s: {tokens_per_sec:,.0f}")
            last_log_time = current_time
            last_log_step = step
            model.train()


def main():
    # setup distributed training
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    
    if is_main_process():
        print(f"running on {world_size} GPUs")
    
    # datasets
    train_dataset = GPTDataset(filename="train.bin", max_seq_len=MAX_SEQ_LEN)
    test_dataset  = GPTDataset(filename="val.bin", max_seq_len=MAX_SEQ_LEN)

    train_sampler = DatasetSampler(len(train_dataset), MICRO_BATCH_SIZE, dist.get_rank(), world_size, shuffle=True)
    val_sampler = DatasetSampler(len(test_dataset), MICRO_BATCH_SIZE, dist.get_rank(), world_size, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(test_dataset, batch_size=MICRO_BATCH_SIZE, sampler=val_sampler, num_workers=4, pin_memory=True)
    
    # model on specific GPU
    model = Model(VOCAB_SIZE, EMBED_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, MAX_SEQ_LEN).to(local_rank)
    model.device = local_rank

    # print total number of parameters
    if is_main_process():
        total_params = count_parameters(model)
        print(f"total parameters: {total_params:,}")
    
    # wrap model with DDP
    model = DDP(model, device_ids=[local_rank])
    
    # synchronize all processes before training
    dist.barrier()
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    
    # train
    train(model, optimizer, train_loader, val_loader, local_rank, world_size)
    
    # save model (only on main process)
    if is_main_process():
        torch.save(model.module.state_dict(), "model.pt")
        print("Model saved to model.pt")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()