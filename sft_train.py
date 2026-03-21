import os
import math
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from model import Model
from data import FineWebDataset
from sft_data import SFTMixtureDataset, prepare_sft_data
from train_utils import count_parameters, validate, get_all_gpu_stats
from dist_utils import setup_distributed, cleanup_distributed, is_main_process
from config import ModelConfig, PostTrainConfig, Paths

import wandb

def train_loop(
    model, optimizer, train_loader, val_loader,
    local_rank, world_size,
    total_steps, warmup_steps,
    lr, min_lr, max_grad_norm, accum_steps,
    tokens_per_step, phase_name, sft=False,
    step_offset=0,
):
    """
    Unified training loop for both mid-training and SFT.
    Matches train.py speed: BF16 autocast, DDP sync overlap, chunked CE.
    """
    model.train()
    step = 0
    last_log_time = time.time()
    last_log_step = 0
    train_iter = iter(train_loader)
    total_tokens = 0

    if is_main_process():
        print(f"[{phase_name}] training for {total_steps:,} steps")
        print(f"[{phase_name}] tokens per step: {tokens_per_step:,}")
        print(f"[{phase_name}] warmup steps: {warmup_steps}")

    while step < total_steps:
        optimizer.zero_grad()
        accum_loss = 0.0
        step_start = time.time()

        for micro_step in range(accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            if sft:
                input_ids, target_ids, loss_mask = batch
                input_ids  = input_ids.to(local_rank, non_blocking=True)
                target_ids = target_ids.to(local_rank, non_blocking=True)
                loss_mask  = loss_mask.to(local_rank, non_blocking=True)
            else:
                input_ids, target_ids = batch
                input_ids  = input_ids.to(local_rank, non_blocking=True)
                target_ids = target_ids.to(local_rank, non_blocking=True)
                loss_mask  = None

            # DDP sync overlap: suppress allreduce on non-final micro-steps
            ctx = model.no_sync() if micro_step < accum_steps - 1 else nullcontext()
            with ctx:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    _, loss = model(input_ids, target_ids, loss_mask=loss_mask)
                scaled = loss / accum_steps
                scaled.backward()

            accum_loss += loss.item()

        avg_loss = accum_loss / accum_steps
        step += 1
        total_tokens += tokens_per_step

        if step < warmup_steps:
            coeff = step / warmup_steps
            current_lr = coeff * lr
        else:
            coeff = 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
            current_lr = min_lr + coeff * (lr - min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        if is_main_process():
            step_elapsed = time.time() - step_start
            tps = tokens_per_step / step_elapsed if step_elapsed > 0 else 0
            ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')

            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            wandb.log({
                f"{phase_name}/loss": avg_loss,
                f"{phase_name}/perplexity": ppl,
                f"{phase_name}/learning_rate": current_lr,
                f"{phase_name}/tokens_per_sec": tps,
                f"{phase_name}/step_time_sec": step_elapsed,
                f"{phase_name}/total_tokens": total_tokens,
                f"{phase_name}/grad_norm": grad_norm,
                f"{phase_name}/progress_pct": (step / total_steps) * 100,
            }, step=step_offset + step)

        if step == 1 and is_main_process():
            print(f"[{phase_name}] first step done, loss: {avg_loss:.4f}", flush=True)

        if step % 100 == 0:
            val_loss = validate(model, val_loader, local_rank, val_steps=50, sft=sft)
            if is_main_process():
                now = time.time()
                elapsed = now - last_log_time
                steps_done = step - last_log_step if step > 0 else 1
                avg_tps = (steps_done * tokens_per_step) / elapsed if elapsed > 0 else 0
                print(f"[{phase_name}] step {step}/{total_steps}  "
                      f"train_loss={avg_loss:.4f}  val_loss={val_loss:.4f}  "
                      f"lr={current_lr:.2e}  tok/s={avg_tps:,.0f}")
                last_log_time = now
                last_log_step = step

                val_ppl = math.exp(val_loss) if val_loss < 20 else float('inf')
                log_dict = {
                    f"{phase_name}/val_loss": val_loss,
                    f"{phase_name}/val_perplexity": val_ppl,
                }
                log_dict.update(get_all_gpu_stats(world_size))
                wandb.log(log_dict, step=step_offset + step)

            model.train()


def main():
    torch.set_float32_matmul_precision('high')

    model_cfg = ModelConfig()
    pt_cfg    = PostTrainConfig()
    paths     = Paths()

    local_rank = setup_distributed()
    world_size = dist.get_world_size()

    if is_main_process():
        print(f"Post-training on {world_size} GPUs")

    if is_main_process():
        prepare_sft_data(paths.sft_data_dir)
    dist.barrier()

    model = Model(
        model_cfg.vocab_size, model_cfg.embed_dim, model_cfg.n_layers,
        model_cfg.n_heads, model_cfg.n_kv_heads, model_cfg.hidden_dim,
        model_cfg.max_seq_len,
    )
    if os.path.exists(paths.pretrained_path):
        state = torch.load(paths.pretrained_path, map_location="cpu", weights_only=True)
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        if is_main_process():
            print(f"Loaded pretrained weights from {paths.pretrained_path}")
    else:
        if is_main_process():
            print("WARNING: no pretrained weights found, training from scratch")

    model = model.to(local_rank)
    model.device = local_rank
    torch._dynamo.config.optimize_ddp = False
    model = torch.compile(model, mode="max-autotune-no-cudagraphs")

    if is_main_process():
        total_params = count_parameters(model)
        print(f"Total parameters: {total_params:,}")

    if is_main_process():
        wandb.init(
            project="apollo-slm",
            name=f"posttrain-{count_parameters(model) // 1_000_000}M-{world_size}gpu",
            config={
                "model/total_params": count_parameters(model),
                "model/max_seq_len": model_cfg.max_seq_len,
                "mid/token_budget": pt_cfg.mid_token_budget,
                "mid/lr": pt_cfg.mid_lr,
                "mid/micro_batch": pt_cfg.mid_micro_batch_size,
                "mid/accum_steps": pt_cfg.mid_accum_steps,
                "sft/token_budget": pt_cfg.sft_token_budget,
                "sft/lr": pt_cfg.sft_lr,
                "sft/micro_batch": pt_cfg.sft_micro_batch_size,
                "sft/accum_steps": pt_cfg.sft_accum_steps,
                "infra/world_size": world_size,
                "infra/gpu_type": torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else "N/A",
            },
        )

    model = DDP(model, device_ids=[local_rank])
    dist.barrier()


    mid_eff_batch   = pt_cfg.mid_micro_batch_size * pt_cfg.mid_accum_steps * world_size
    mid_tok_per_step = mid_eff_batch * model_cfg.max_seq_len
    mid_total_steps = pt_cfg.mid_token_budget // mid_tok_per_step

    if os.path.exists(paths.midtrained_path):
        state = torch.load(paths.midtrained_path, map_location="cpu", weights_only=True)
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.module._orig_mod.load_state_dict(state)
        if is_main_process():
            print(f"Loaded mid-trained model from {paths.midtrained_path}, skipping mid-training")
        dist.barrier()
    else:
        if is_main_process():
            print("\n" + "=" * 60)
            print("  PHASE 1: Mid-Training (FineWeb-EDU)")
            print("=" * 60)

        mid_train_dataset = FineWebDataset(
            manifest_path=paths.pretrain_train_manifest_path,
            max_seq_len=model_cfg.max_seq_len, seed=42, shuffle=True,
        )
        mid_val_dataset = FineWebDataset(
            manifest_path=paths.pretrain_val_manifest_path,
            max_seq_len=model_cfg.max_seq_len, seed=42, shuffle=False,
        )
        mid_train_loader = DataLoader(
            mid_train_dataset, batch_size=pt_cfg.mid_micro_batch_size,
            num_workers=8, pin_memory=True,
        )
        mid_val_loader = DataLoader(
            mid_val_dataset, batch_size=pt_cfg.mid_micro_batch_size,
            num_workers=8, pin_memory=True,
        )

        mid_warmup = int(pt_cfg.mid_warmup_ratio * mid_total_steps)

        mid_optimizer = torch.optim.AdamW(
            model.parameters(), lr=pt_cfg.mid_lr,
            weight_decay=pt_cfg.mid_weight_decay, fused=True,
        )

        train_loop(
            model, mid_optimizer, mid_train_loader, mid_val_loader,
            local_rank, world_size,
            total_steps=mid_total_steps, warmup_steps=mid_warmup,
            lr=pt_cfg.mid_lr, min_lr=pt_cfg.mid_min_lr,
            max_grad_norm=pt_cfg.max_grad_norm,
            accum_steps=pt_cfg.mid_accum_steps,
            tokens_per_step=mid_tok_per_step,
            phase_name="mid", sft=False,
        )

        if is_main_process():
            os.makedirs(os.path.dirname(paths.midtrained_path), exist_ok=True)
            torch.save(model.module.state_dict(), paths.midtrained_path)
            print(f"Mid-trained model saved to {paths.midtrained_path}")

        dist.barrier()


    if is_main_process():
        print("\n" + "=" * 60)
        print("  PHASE 2: SFT (SmolTalk + MMLU + GSM8K)")
        print("=" * 60)

    sft_train_dataset = SFTMixtureDataset(
        data_path=paths.sft_train_path,
        max_seq_len=model_cfg.max_seq_len,
        rank=dist.get_rank(), world_size=world_size,
        seed=42, shuffle=True,
    )
    sft_val_dataset = SFTMixtureDataset(
        data_path=paths.sft_val_path,
        max_seq_len=model_cfg.max_seq_len,
        rank=dist.get_rank(), world_size=world_size,
        seed=42, shuffle=False,
    )
    sft_train_loader = DataLoader(
        sft_train_dataset, batch_size=pt_cfg.sft_micro_batch_size,
        num_workers=8, pin_memory=True,
    )
    sft_val_loader = DataLoader(
        sft_val_dataset, batch_size=pt_cfg.sft_micro_batch_size,
        num_workers=8, pin_memory=True,
    )

    sft_eff_batch   = pt_cfg.sft_micro_batch_size * pt_cfg.sft_accum_steps * world_size
    sft_tok_per_step = sft_eff_batch * model_cfg.max_seq_len
    sft_total_steps = pt_cfg.sft_token_budget // sft_tok_per_step
    sft_warmup      = int(pt_cfg.sft_warmup_ratio * sft_total_steps)


    sft_optimizer = torch.optim.AdamW(
        model.parameters(), lr=pt_cfg.sft_lr,
        weight_decay=pt_cfg.sft_weight_decay, fused=True,
    )

    train_loop(
        model, sft_optimizer, sft_train_loader, sft_val_loader,
        local_rank, world_size,
        total_steps=sft_total_steps, warmup_steps=sft_warmup,
        lr=pt_cfg.sft_lr, min_lr=pt_cfg.sft_min_lr,
        max_grad_norm=pt_cfg.max_grad_norm,
        accum_steps=pt_cfg.sft_accum_steps,
        tokens_per_step=sft_tok_per_step,
        phase_name="sft", sft=True,
        step_offset=mid_total_steps,
    )

    # save final SFT model
    if is_main_process():
        os.makedirs(os.path.dirname(paths.sft_path), exist_ok=True)
        torch.save(model.module.state_dict(), paths.sft_path)
        print(f"SFT model saved to {paths.sft_path}")

        artifact = wandb.Artifact("apollo-sft", type="model")
        artifact.add_file(paths.sft_path)
        wandb.log_artifact(artifact)
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
