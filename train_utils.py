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

