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
    micro_batch_size = 24
    accum_steps = 8
    lr = 3e-4
    min_lr = 3e-5
    token_budget = 10_000_000_000
    warmrup_ratio = 0.1
    weight_decay = 0.1
    max_grad_norm = 1.0

@dataclass
class SFTConfig:
    micro_batch_size = 8
    accum_steps = 4 
    num_epochs = 3
    lr = 2e-3 
    min_lr = 2e-6
    warmup_ratio = 0.1
    mask_prompt = True

@dataclass
class Paths:
    pretrained_path = "models/apollo_pretrained.pt"
    pretrain_train_data_path = "data/train.bin"
    pretrain_val_data_path = "data/val.bin"
    sft_path = "apollo_sft.pt"
    sft_data_path = "alpaca.json"

