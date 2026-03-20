from dataclasses import dataclass

@dataclass
class ModelConfig:
    # 367m config 
    vocab_size = 100288 
    embed_dim = 1024
    n_layers = 28
    n_heads = 16
    n_kv_heads = 8
    hidden_dim = 2048
    max_seq_len = 1024


@dataclass
class TrainConfig:
    micro_batch_size = 48
    accum_steps = 6
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
class PostTrainConfig:
    # Phase 1: Mid-training (continued pretraining on high-quality data)
    mid_micro_batch_size = 48
    mid_accum_steps = 6
    mid_lr = 1e-4
    mid_min_lr = 1e-5
    mid_token_budget = 2_000_000_000   
    mid_warmup_ratio = 0.05
    mid_weight_decay = 0.1

    # Phase 2: SFT (supervised fine-tuning on conversations)
    sft_micro_batch_size = 16
    sft_accum_steps = 2
    sft_lr = 5e-5
    sft_min_lr = 5e-6
    sft_token_budget = 200_000_000    
    sft_warmup_ratio = 0.05
    sft_weight_decay = 0.01

    # Shared
    max_grad_norm = 1.0

@dataclass
class Paths:
    pretrained_path = "models/apollo_pretrained.pt"
    midtrained_path = "models/apollo_midtrained.pt"
    sft_path = "models/apollo_sft.pt"
    pretrain_train_manifest_path = "data/fineweb_edu_train.jsonl"
    pretrain_val_manifest_path = "data/fineweb_edu_val.jsonl"
    sft_data_dir = "data/sft"
    sft_train_path = "data/sft/sft_train.jsonl"
    sft_val_path = "data/sft/sft_val.jsonl"