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

# model config (must match pretrained model)
VOCAB_SIZE = 50257
EMBED_DIM = 1024
N_LAYERS = 20
N_HEADS = 16
N_KV_HEADS = 8
HIDDEN_DIM = 2048
MAX_SEQ_LEN = 1024

# sft training config
MICRO_BATCH_SIZE = 8
ACCUM_STEPS = 4
NUM_EPOCHS = 3

# learning rate (lower than pretraining)
LR = 2e-5
MIN_LR = 2e-6
WARMUP_RATIO = 0.03

# paths
PRETRAINED_PATH = "model.pt"
SFT_OUTPUT_PATH = "model_sft.pt"
DATA_PATH = "alpaca.json"

# whether to mask loss on prompt tokens
MASK_PROMPT = True


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


def masked_loss(logits, targets, loss_mask):
    """Compute cross-entropy loss with masking."""
    B, T, C = logits.shape
    
    # flatten for cross entropy
    logits_flat = logits.view(B * T, C)
    targets_flat = targets.view(B * T)
    loss_mask_flat = loss_mask.view(B * T)
    
    # compute per-token loss
    loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    
    # apply mask and average
    masked_loss = (loss_per_token * loss_mask_flat).sum() / (loss_mask_flat.sum() + 1e-8)
    
    return masked_loss


def validate(model, val_loader, local_rank, num_steps=50):
    model.eval()
    val_loss = 0.0
    val_iter = iter(val_loader)
    
    with torch.no_grad():
        for _ in range(num_steps):
            try:
                input_ids, target_ids, loss_mask = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                input_ids, target_ids, loss_mask = next(val_iter)
            
            input_ids = input_ids.to(local_rank)
            target_ids = target_ids.to(local_rank)
            loss_mask = loss_mask.to(local_rank)
            
            logits, _ = model(input_ids)
            loss = masked_loss(logits, target_ids, loss_mask)
            val_loss += loss.item()
    
    return val_loss / num_steps


def train(model, optimizer, train_loader, val_loader, local_rank, world_size, total_steps):
    model.train()
    effective_batch_size = MICRO_BATCH_SIZE * ACCUM_STEPS * world_size
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    if is_main_process():
        print(f"sft training for {total_steps:,} steps")
        print(f"effective batch size: {effective_batch_size}")
        print(f"warmup steps: {warmup_steps}")
        print(f"mask prompt tokens: {MASK_PROMPT}")
    
    step = 0
    epoch = 0
    last_log_time = time.time()
    train_iter = iter(train_loader)
    
    while step < total_steps:
        optimizer.zero_grad()
        accum_loss = 0.0
        
        # gradient accumulation
        for micro_step in range(ACCUM_STEPS):
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
            if micro_step < ACCUM_STEPS - 1:
                with model.no_sync():
                    logits, _ = model(input_ids)
                    loss = masked_loss(logits, target_ids, loss_mask)
                    scaled_loss = loss / ACCUM_STEPS
                    scaled_loss.backward()
            else:
                logits, _ = model(input_ids)
                loss = masked_loss(logits, target_ids, loss_mask)
                scaled_loss = loss / ACCUM_STEPS
                scaled_loss.backward()
            
            accum_loss += loss.item()
        
        avg_loss = accum_loss / ACCUM_STEPS
        step += 1
        
        # learning rate schedule (warmup + cosine decay)
        if step < warmup_steps:
            lr = LR * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = MIN_LR + 0.5 * (LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))
        
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
    if is_main_process() and not os.path.exists(DATA_PATH):
        download_alpaca()
    dist.barrier()
    
    # load datasets
    full_dataset = SFTDataset(DATA_PATH, MAX_SEQ_LEN, mask_prompt=MASK_PROMPT)
    
    # split into train/val (95/5)
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * 0.05)
    train_size = dataset_size - val_size
    
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, dataset_size))
    
    if is_main_process():
        print(f"train examples: {len(train_dataset)}, val examples: {len(val_dataset)}")
    
    # calculate total steps
    steps_per_epoch = len(train_dataset) // (MICRO_BATCH_SIZE * ACCUM_STEPS * world_size)
    total_steps = steps_per_epoch * NUM_EPOCHS
    
    # samplers and loaders
    train_sampler = SFTSampler(len(train_dataset), MICRO_BATCH_SIZE, dist.get_rank(), world_size, shuffle=True)
    val_sampler = SFTSampler(len(val_dataset), MICRO_BATCH_SIZE, dist.get_rank(), world_size, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=MICRO_BATCH_SIZE, sampler=val_sampler, num_workers=4, pin_memory=True)
    
    # load pretrained model
    model = Model(VOCAB_SIZE, EMBED_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, MAX_SEQ_LEN)
    
    if os.path.exists(PRETRAINED_PATH):
        state_dict = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        if is_main_process():
            print(f"loaded pretrained weights from {PRETRAINED_PATH}")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    # train
    train(model, optimizer, train_loader, val_loader, local_rank, world_size, total_steps)
    
    # save final model
    if is_main_process():
        torch.save(model.module.state_dict(), SFT_OUTPUT_PATH)
        print(f"sft model saved to {SFT_OUTPUT_PATH}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()