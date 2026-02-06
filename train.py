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
from train_utils import *
from dist_utils import *
from config import *


def train(model, optimizer, train_loader, val_loader, local_rank, world_size, model_cfg, train_cfg):
    model.train()
    total_steps = num_steps(train_cfg, model_cfg, world_size)
    effective_batch_size = train_cfg.micro_batch_size * train_cfg.accum_steps * world_size
    tokens_per_step = effective_batch_size * model_cfg.max_seq_len
    warmup_steps = int(train_cfg.warmrup_ratio * total_steps)
    
    if is_main_process():
        print(f"training for {total_steps:,} steps")
        print(f"effective batch size: {effective_batch_size}")
        print(f"warmup steps: {warmup_steps}")
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
        for micro_step in range(train_cfg.accum_steps):
            try:
                input, target = next(train_iter)
            except StopIteration:
                train_loader.sampler.set_epoch(step)
                train_iter = iter(train_loader)
                input, target = next(train_iter)
            
            input = input.to(local_rank)
            target = target.to(local_rank)
            
            # disable gradient sync for all but the last micro step
            if micro_step < train_cfg.accum_steps - 1:
                with model.no_sync():
                    out, loss = model(input, target)
                    scaled_loss = loss / train_cfg.accum_steps
                    scaled_loss.backward()
            else:
                out, loss = model(input, target)
                scaled_loss = loss / train_cfg.accum_steps
                scaled_loss.backward()
            
            accum_loss += loss.item()
        
        # average loss over accumulation steps
        avg_loss = accum_loss / train_cfg.accum_steps
        
        step += 1
        
        # learning rate schedule
        if step < warmup_steps:
            coeff = step / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = coeff * train_cfg.lr
        else:
            coeff = 0.5 * (1.0 + math.cos(math.pi * ((step - train_cfg.warmup_steps) / (total_steps - train_cfg.warmup_steps))))
            for param_group in optimizer.param_groups:
                param_group['lr'] = train_cfg.min_lr + coeff * (train_cfg.lr - train_cfg.min_lr)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.max_grad_norm)
        optimizer.step()
        
        if step == 1 and is_main_process():
            print(f"first step completed, loss: {avg_loss:.4f}", flush=True)

        if step % 100 == 0 and is_main_process():
            val_loss = validate(model, val_loader, val_steps=50, local_rank=local_rank, sft=False)
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
    # configs
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    paths_cfg = Paths()
    
    # setup distributed training
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    
    if is_main_process():
        print(f"running on {world_size} GPUs")
    
    # datasets
    train_dataset = GPTDataset(filename=paths_cfg.pretrain_train_data_path, max_seq_len=model_cfg.max_seq_len)
    test_dataset  = GPTDataset(filename=paths_cfg.pretrain_val_data_path, max_seq_len=model_cfg.max_seq_len)

    train_sampler = DatasetSampler(len(train_dataset), train_cfg.micro_batch_size, dist.get_rank(), world_size, shuffle=True)
    val_sampler = DatasetSampler(len(test_dataset), train_cfg.micro_batch_size, dist.get_rank(), world_size, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.micro_batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(test_dataset, batch_size=train_cfg.micro_batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)
    
    # model on specific GPU
    model = Model(model_cfg.vocab_size, model_cfg.embed_dim, model_cfg.n_layers, model_cfg.n_heads, model_cfg.n_kv_heads, model_cfg.hidden_dim, model_cfg.max_seq_len).to(local_rank)
    model.device = local_rank
    model = torch.compile(model, mode="default")

    # print total number of parameters
    if is_main_process():
        total_params = count_parameters(model)
        print(f"total parameters: {total_params:,}")
    
    # wrap model with DDP
    model = DDP(model, device_ids=[local_rank])
    
    # synchronize all processes before training
    dist.barrier()
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, fused=True)
    
    # train
    train(model, optimizer, train_loader, val_loader, local_rank, world_size, model_cfg, train_cfg)
    
    # save model (only on main process)
    if is_main_process():
        torch.save(model.module.state_dict(), paths_cfg.pretrained_path)
        print("Model saved")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()