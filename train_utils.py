import torch.nn.functional as F 
from config import *
import torch
from torch.amp import autocast

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def num_steps(train_cfg: TrainConfig, model_cfg: ModelConfig, world_size: int):
    effective_batch_size = train_cfg.micro_batch_size * train_cfg.accum_steps * world_size
    total_steps = train_cfg.token_budget // (effective_batch_size * model_cfg.max_seq_len)
    return total_steps


def masked_loss(logits, targets, loss_mask):
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


def validate(model, val_loader, local_rank, val_steps=20, sft=False):
    model.eval() 
    val_loss = 0.0 
    val_iter = iter(val_loader)

    with torch.no_grad():
        for _ in range(val_steps):
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                batch = next(val_iter)

            if sft:
                input_ids, target_ids, loss_mask = batch 
                input_ids = input_ids.to(local_rank)
                target_ids = target_ids.to(local_rank)
                loss_mask = loss_mask.to(local_rank)

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, _ = model(input_ids)
                loss = masked_loss(logits, target_ids, loss_mask)
                val_loss += loss.item()
            
            else:
                input_ids, target_ids = batch
                input_ids = input_ids.to(local_rank)
                target_ids = target_ids.to(local_rank)

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, _ = model(input_ids)
                
                loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), target_ids.view(-1))
                val_loss += loss.item()

    return val_loss / val_steps


# gpu monitoring utilities for wandb logging
import subprocess

def get_gpu_stats(device_index=0):
    stats = {}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            stats["gpu/utilization_pct"] = float(parts[0])
            stats["gpu/temperature_C"] = float(parts[1])
            stats["gpu/memory_used_MB"] = float(parts[2])
            stats["gpu/memory_total_MB"] = float(parts[3])
            stats["gpu/memory_used_pct"] = float(parts[2]) / float(parts[3]) * 100
            stats["gpu/power_draw_W"] = float(parts[4])
            stats["gpu/power_limit_W"] = float(parts[5])
    except Exception:
        pass

    # pytorch-level cuda memory stats
    if torch.cuda.is_available():
        stats["gpu/torch_memory_allocated_MB"] = torch.cuda.memory_allocated(device_index) / 1e6
        stats["gpu/torch_memory_reserved_MB"] = torch.cuda.memory_reserved(device_index) / 1e6
        stats["gpu/torch_max_memory_allocated_MB"] = torch.cuda.max_memory_allocated(device_index) / 1e6

    return stats


def get_all_gpu_stats(world_size):
    all_stats = {}
    for i in range(world_size):
        per_gpu = get_gpu_stats(i)
        for k, v in per_gpu.items():
            all_stats[f"{k}/gpu_{i}"] = v
    # log averages across gpus
    keys_to_avg = ["gpu/utilization_pct", "gpu/temperature_C", "gpu/memory_used_pct", "gpu/power_draw_W"]
    for key in keys_to_avg:
        vals = [get_gpu_stats(i).get(key) for i in range(world_size)]
        vals = [v for v in vals if v is not None]
        if vals:
            all_stats[f"{key}/avg"] = sum(vals) / len(vals)
    return all_stats