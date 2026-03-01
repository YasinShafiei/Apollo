from dataclasses import dataclass

@dataclass
class ModelConfig:
    # 367m config 
    vocab_size = 100277
    embed_dim = 1024
    n_layers = 28
    n_heads = 16
    n_kv_heads = 8
    hidden_dim = 2048
    max_seq_len = 1024


@dataclass
class TrainConfig:
    micro_batch_size = 24
    accum_steps = 12
    lr = 3e-4
    min_lr = 3e-5
    token_budget = 30_000_000_000
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
    pretrain_train_manifest_path = "data/fineweb_edu_train.jsonl"
    pretrain_val_manifest_path = "data/fineweb_edu_val.jsonl"
    sft_path = "apollo_sft.pt"
    sft_data_path = "alpaca.json"

