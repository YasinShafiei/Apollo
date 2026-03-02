import os
import sys
import math
import time

import torch 
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from model import Model
from data import FineWebDataset
from train_utils import *
from dist_utils import *
from config import *

import wandb


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
    total_tokens_processed = 0
    
    if is_main_process():
        print("starting training loop", flush=True)
    
    while step < total_steps:
        optimizer.zero_grad()
        accum_loss = 0.0
        step_start_time = time.time()
        
        # gradient accumulation loop
        for micro_step in range(train_cfg.accum_steps):
            try:
                input, target = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input, target = next(train_iter)
            
            input = input.to(local_rank)
            target = target.to(local_rank)
            
            # always disable DDP gradient sync during backward to keep
            # the backward graph structure consistent for CUDA graphs
            # (NCCL allreduce is not capturable by CUDA graphs)
            with model.no_sync():
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    out, loss = model(input, target)
                scaled_loss = loss / train_cfg.accum_steps
                scaled_loss.backward()
            
            accum_loss += loss.item()

        # manually synchronize gradients across all ranks after
        # accumulation, outside the compiled backward graph
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        
        # average loss over accumulation steps
        avg_loss = accum_loss / train_cfg.accum_steps
        
        step += 1
        total_tokens_processed += tokens_per_step
        
        # learning rate schedule
        if step < warmup_steps:
            coeff = step / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = coeff * train_cfg.lr
        else:
            coeff = 0.5 * (1.0 + math.cos(math.pi * ((step - warmup_steps) / (total_steps - warmup_steps))))
            for param_group in optimizer.param_groups:
                param_group['lr'] = train_cfg.min_lr + coeff * (train_cfg.lr - train_cfg.min_lr)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.max_grad_norm)
        optimizer.step()

        # wandb per-step metrics
        if is_main_process():
            step_elapsed = time.time() - step_start_time
            step_tokens_per_sec = tokens_per_step / step_elapsed if step_elapsed > 0 else 0
            current_lr = optimizer.param_groups[0]['lr']
            perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

            # post-clip gradient norm
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            wandb.log({
                "train/loss": avg_loss,
                "train/perplexity": perplexity,
                "train/learning_rate": current_lr,
                "train/tokens_per_sec": step_tokens_per_sec,
                "train/step_time_sec": step_elapsed,
                "train/total_tokens": total_tokens_processed,
                "train/grad_norm": grad_norm,
                "train/progress_pct": (step / total_steps) * 100,
            }, step=step)
        
        if step == 1 and is_main_process():
            print(f"first step completed, loss: {avg_loss:.4f}", flush=True)

        if step % 100 == 0:
            val_loss = validate(model, val_loader, val_steps=50, local_rank=local_rank, sft=False)
            if is_main_process():
                current_time = time.time()
                elapsed = current_time - last_log_time
                steps_since_last_log = step - last_log_step if step > 0 else 1
                tokens_per_sec = (steps_since_last_log * tokens_per_step) / elapsed if elapsed > 0 else 0
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Step {step}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}, Tokens/s: {tokens_per_sec:,.0f}")
                last_log_time = current_time
                last_log_step = step

                # wandb validation + gpu stats
                val_perplexity = math.exp(val_loss) if val_loss < 20 else float('inf')
                log_dict = {
                    "val/loss": val_loss,
                    "val/perplexity": val_perplexity,
                    "train/tokens_per_sec_avg": tokens_per_sec,
                }
                gpu_stats = get_all_gpu_stats(world_size)
                log_dict.update(gpu_stats)
                wandb.log(log_dict, step=step)

            model.train()

def main():
    torch.set_float32_matmul_precision('high')

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
    train_dataset = FineWebDataset(
        manifest_path=paths_cfg.pretrain_train_manifest_path,
        max_seq_len=model_cfg.max_seq_len,
        seed=42,
        shuffle=True,
    )
    val_dataset = FineWebDataset(
        manifest_path=paths_cfg.pretrain_val_manifest_path,
        max_seq_len=model_cfg.max_seq_len,
        seed=42,
        shuffle=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=train_cfg.micro_batch_size, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=train_cfg.micro_batch_size, num_workers=4, pin_memory=True)
    
    # model on specific GPU with BF16
    model = Model(model_cfg.vocab_size, model_cfg.embed_dim, model_cfg.n_layers, model_cfg.n_heads, model_cfg.n_kv_heads, model_cfg.hidden_dim, model_cfg.max_seq_len).to(local_rank)
    model.device = local_rank
    model = torch.compile(model, mode="max-autotune")

    # print total number of parameters
    if is_main_process():
        total_params = count_parameters(model)
        print(f"total parameters: {total_params:,}")
    
    # wandb init (only on main process)
    if is_main_process():
        wandb.init(
            project="apollo-slm",
            name=f"pretrain-{total_params // 1_000_000}M-{world_size}gpu",
            config={
                "model/vocab_size": model_cfg.vocab_size,
                "model/embed_dim": model_cfg.embed_dim,
                "model/n_layers": model_cfg.n_layers,
                "model/n_heads": model_cfg.n_heads,
                "model/n_kv_heads": model_cfg.n_kv_heads,
                "model/hidden_dim": model_cfg.hidden_dim,
                "model/max_seq_len": model_cfg.max_seq_len,
                "model/total_params": total_params,
                "train/micro_batch_size": train_cfg.micro_batch_size,
                "train/accum_steps": train_cfg.accum_steps,
                "train/effective_batch_size": train_cfg.micro_batch_size * train_cfg.accum_steps * world_size,
                "train/lr": train_cfg.lr,
                "train/min_lr": train_cfg.min_lr,
                "train/token_budget": train_cfg.token_budget,
                "train/warmup_ratio": train_cfg.warmrup_ratio,
                "train/weight_decay": train_cfg.weight_decay,
                "train/max_grad_norm": train_cfg.max_grad_norm,
                "infra/world_size": world_size,
                "infra/gpu_type": torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else "N/A",
            },
        )
        wandb.watch(model, log="gradients", log_freq=500)

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

        # save model as wandb artifact
        artifact = wandb.Artifact("apollo-pretrained", type="model")
        artifact.add_file(paths_cfg.pretrained_path)
        wandb.log_artifact(artifact)
        wandb.finish()
    
    cleanup_distributed()


if __name__ == "__main__":
    main()