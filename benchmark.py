import os
import time
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import Model
from train_utils import count_parameters
from dist_utils import setup_distributed, cleanup_distributed, is_main_process
from config import ModelConfig, TrainConfig


def run_benchmark(model, device, model_cfg, train_cfg, num_steps=100, warmup_steps=10):
    """Run benchmark and return tokens per second."""
    model.train()
    
    # create synthetic data
    batch_size = train_cfg.micro_batch_size
    seq_len = model_cfg.max_seq_len
    vocab_size = model_cfg.vocab_size
    
    # pre-generate batches to avoid data generation overhead
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    
    # warmup
    if is_main_process():
        print(f"warming up for {warmup_steps} steps...")
    
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        _, loss = model(input_ids, target_ids)
        loss.backward()
        optimizer.step()
    
    # synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    
    # benchmark
    if is_main_process():
        print(f"benchmarking for {num_steps} steps...")
    
    start_time = time.perf_counter()
    
    for step in range(num_steps):
        optimizer.zero_grad()
        _, loss = model(input_ids, target_ids)
        loss.backward()
        optimizer.step()
    
    # synchronize after timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    return elapsed, num_steps


def print_results(elapsed, num_steps, model_cfg, train_cfg, world_size):
    """Print benchmark results."""
    tokens_per_step = train_cfg.micro_batch_size * model_cfg.max_seq_len * world_size
    total_tokens = tokens_per_step * num_steps
    tokens_per_sec = total_tokens / elapsed
    steps_per_sec = num_steps / elapsed
    ms_per_step = (elapsed / num_steps) * 1000
    
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"steps:              {num_steps}")
    print(f"total time:         {elapsed:.2f}s")
    print(f"steps/sec:          {steps_per_sec:.2f}")
    print(f"ms/step:            {ms_per_step:.2f}")
    print(f"tokens/sec:         {tokens_per_sec:,.0f}")
    print(f"tokens/sec/gpu:     {tokens_per_sec / world_size:,.0f}")
    print("=" * 50)
    
    # memory stats
    if torch.cuda.is_available():
        print("\nMEMORY USAGE")
        print("-" * 50)
        for i in range(world_size):
            if i == dist.get_rank() if dist.is_initialized() else 0:
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def main_single_gpu(args):
    """Single GPU benchmark."""
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running single GPU benchmark on {device}")
    
    # model
    model = Model(
        model_cfg.vocab_size,
        model_cfg.embed_dim,
        model_cfg.n_layers,
        model_cfg.n_heads,
        model_cfg.n_kv_heads,
        model_cfg.hidden_dim,
        model_cfg.max_seq_len
    ).to(device)
    
    if args.compile:
        print("compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
    
    print(f"total parameters: {count_parameters(model):,}")
    print(f"batch size: {train_cfg.micro_batch_size}")
    print(f"sequence length: {model_cfg.max_seq_len}")
    
    elapsed, num_steps = run_benchmark(
        model, device, model_cfg, train_cfg,
        num_steps=args.steps, warmup_steps=args.warmup
    )
    
    print_results(elapsed, num_steps, model_cfg, train_cfg, world_size=1)


def main_distributed(args):
    """Distributed benchmark."""
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    
    if is_main_process():
        print(f"running distributed benchmark on {world_size} GPUs")
    
    # model
    model = Model(
        model_cfg.vocab_size,
        model_cfg.embed_dim,
        model_cfg.n_layers,
        model_cfg.n_heads,
        model_cfg.n_kv_heads,
        model_cfg.hidden_dim,
        model_cfg.max_seq_len
    ).to(local_rank)
    
    if args.compile:
        if is_main_process():
            print("compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
    
    model = DDP(model, device_ids=[local_rank])
    
    if is_main_process():
        print(f"total parameters: {count_parameters(model):,}")
        print(f"micro batch size: {train_cfg.micro_batch_size}")
        print(f"effective batch size: {train_cfg.micro_batch_size * world_size}")
        print(f"sequence length: {model_cfg.max_seq_len}")
    
    dist.barrier()
    
    elapsed, num_steps = run_benchmark(
        model, local_rank, model_cfg, train_cfg,
        num_steps=args.steps, warmup_steps=args.warmup
    )
    
    if is_main_process():
        print_results(elapsed, num_steps, model_cfg, train_cfg, world_size)
    
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Apollo pretraining speed")
    parser.add_argument("--steps", type=int, default=100, help="number of benchmark steps")
    parser.add_argument("--warmup", type=int, default=10, help="number of warmup steps")
    parser.add_argument("--compile", action="store_true", help="use torch.compile")
    args = parser.parse_args()
    
    # check if running with torchrun (distributed)
    if "LOCAL_RANK" in os.environ:
        main_distributed(args)
    else:
        main_single_gpu(args)


if __name__ == "__main__":
    main()