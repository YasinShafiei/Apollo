
from config import *
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def num_steps(train_cfg: TrainConfig, model_cfg: ModelConfig, world_size: int):
    effective_batch_size = train_cfg.micro_batch_size * train_cfg.accum_steps * world_size
    total_steps = train_cfg.token_budget // (effective_batch_size * model_cfg.max_seq_len)
    return total_steps

def validate(model, val_loader, val_steps=20):
    model.eval()
    val_loss = 0.0
    val_iter = iter(val_loader)
    with torch.no_grad():
        for _ in range(val_steps):
            try:
                input, target = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                input, target = next(val_iter)
            input = input.to(model.device)
            target = target.to(model.device)
            out, loss = model(input, target)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / val_steps
    return avg_val_loss