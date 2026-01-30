import os
import math
import time

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from model import Model
from sft_data import SFTDataset, SFTSampler, download_alpaca
from train_utils import *
from dist_utils import *
from config import ModelConfig, SFTConfig, Paths


def train(model, optimizer, train_loader, val_loader, local_rank, world_size, total_steps):
    model.train()
    effective_batch_size = SFTConfig.micro_batch_size * SFTConfig.accum_steps * world_size
    warmup_steps = int(total_steps * SFTConfig.warmup_ratio)
    
    if is_main_process():
        print(f"sft training for {total_steps:,} steps")
        print(f"effective batch size: {effective_batch_size}")
        print(f"warmup steps: {warmup_steps}")
        print(f"mask prompt tokens: {SFTConfig.mask_prompt}")
    
    step = 0
    epoch = 0
    last_log_time = time.time()
    train_iter = iter(train_loader)
    
    while step < total_steps:
        optimizer.zero_grad()
        accum_loss = 0.0
        
        # gradient accumulation
        for micro_step in range(SFTConfig.accum_steps):
            try:
                input_ids, target_ids, loss_mask = next(train_iter)
            except StopIteration:
                epoch += 1
                train_loader.sampler.set_epoch(epoch)
                train_iter = iter(train_loader)
                input_ids, target_ids, loss_mask = next(train_iter)
                if is_main_process():
                    print(f"starting epoch {epoch + 1}")
            
            input_ids = input_ids.to(local_rank)
            target_ids = target_ids.to(local_rank)
            loss_mask = loss_mask.to(local_rank)
            
            # forward pass
            if micro_step < SFTConfig.accum_steps - 1:
                with model.no_sync():
                    logits, _ = model(input_ids)
                    loss = masked_loss(logits, target_ids, loss_mask)
                    scaled_loss = loss / SFTConfig.accum_steps
                    scaled_loss.backward()
            else:
                logits, _ = model(input_ids)
                loss = masked_loss(logits, target_ids, loss_mask)
                scaled_loss = loss / SFTConfig.accum_steps
                scaled_loss.backward()
            
            accum_loss += loss.item()
        
        avg_loss = accum_loss / SFTConfig.accum_steps
        step += 1
        
        # learning rate schedule (warmup + cosine decay)
        if step < warmup_steps:
            lr = SFTConfig.lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = SFTConfig.min_lr + 0.5 * (SFTConfig.lr - SFTConfig.min_lr) * (1.0 + math.cos(math.pi * progress))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # logging
        if step % 50 == 0 and is_main_process():
            val_loss = validate(model, val_loader, local_rank, num_steps=20)
            elapsed = time.time() - last_log_time
            steps_per_sec = 50 / elapsed
            print(f"step {step}/{total_steps}, train loss: {avg_loss:.4f}, val loss: {val_loss:.4f}, lr: {lr:.2e}, steps/s: {steps_per_sec:.2f}")
            last_log_time = time.time()
            model.train()
        
        # save checkpoint
        if step % 500 == 0 and is_main_process():
            checkpoint_path = f"model_sft_step{step}.pt"
            torch.save(model.module.state_dict(), checkpoint_path)
            print(f"checkpoint saved to {checkpoint_path}")


def main():
    # setup distributed
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    
    if is_main_process():
        print(f"running sft on {world_size} GPUs")
    
    # download data if needed
    if is_main_process() and not os.path.exists(Paths.sft_data_path):
        download_alpaca()
    dist.barrier()
    
    # load datasets
    full_dataset = SFTDataset(Paths.sft_data_path, ModelConfig.max_seq_len, mask_prompt=SFTConfig.mask_prompt)
    
    # split into train/val (95/5)
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * 0.05)
    train_size = dataset_size - val_size
    
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, dataset_size))
    
    if is_main_process():
        print(f"train examples: {len(train_dataset)}, val examples: {len(val_dataset)}")
    
    # calculate total steps
    steps_per_epoch = len(train_dataset) // (SFTConfig.micro_batch_size * SFTConfig.accum_steps * world_size)
    total_steps = steps_per_epoch * SFTConfig.num_epochs
    
    # samplers and loaders
    train_sampler = SFTSampler(len(train_dataset), SFTConfig.micro_batch_size, dist.get_rank(), world_size, shuffle=True)
    val_sampler = SFTSampler(len(val_dataset), SFTConfig.micro_batch_size, dist.get_rank(), world_size, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=SFTConfig.micro_batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=SFTConfig.micro_batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)
    
    # load pretrained model
    model = Model(ModelConfig.vocab_size, ModelConfig.embed_dim, ModelConfig.n_layers, ModelConfig.n_heads, ModelConfig.n_kv_heads, ModelConfig.hidden_dim, ModelConfig.max_seq_len)
    
    if os.path.exists(Paths.pretrained_path):
        state_dict = torch.load(Paths.pretrained_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        if is_main_process():
            print(f"loaded pretrained weights from {Paths.pretrained_path}")
    else:
        if is_main_process():
            print("WARNING: no pretrained weights found, training from scratch")
    
    model = model.to(local_rank)
    model.device = local_rank
    
    if is_main_process():
        print(f"total parameters: {count_parameters(model):,}")
    
    # wrap with DDP
    model = DDP(model, device_ids=[local_rank])
    
    dist.barrier()
    
    # optimizer (lower weight decay for sft)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SFTConfig.lr, weight_decay=0.01)
    
    # train
    train(model, optimizer, train_loader, val_loader, local_rank, world_size, total_steps)
    
    # save final model
    if is_main_process():
        torch.save(model.module.state_dict(), Paths.sft_path)
        print(f"sft model saved to {Paths.sft_path}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()