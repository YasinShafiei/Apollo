from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size = 50257
    embed_dim = 1024
    n_layers = 20
    n_heads = 16
    n_kv_heads = 8
    hidden_dim = 2048
    max_seq_len = 1024


@dataclass
class TrainConfig:
    micro_batch_size = 16
    accum_steps = 4
    lr = 3e-4
    min_lr = 3e-5
    token_budget = 10_000_000_000
    warmup_steps = 1200
    weight_decay = 0.1
    max_grad_norm = 1.0
