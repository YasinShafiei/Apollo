# Apollo SLM

Apollo SLM is a simple, open-source implementation of a transformer-based language model, designed for research and educational purposes. The project includes code for data preparation, model definition, pretraining, supervised fine-tuning (SFT), and inference.

## Features
- Transformer-based architecture (20 layers, 16 heads, rotary embeddings)
- Pretraining on OpenWebText (or similar datasets)
- Supervised fine-tuning (SFT) with instruction-following data (e.g., Alpaca)
- Distributed training support (DDP)
- Inference and chat interface

## Model Architecture
- **Vocabulary Size:** 50,257
- **Embedding Dimension:** 1,024
- **Layers:** 20
- **Attention Heads:** 16 (8 key-value heads)
- **Hidden Dimension:** 2,048
- **Max Sequence Length:** 1,024

## Training
Pretraining and SFT are supported. Distributed training is enabled via PyTorch DDP.

### Example Training Log
```
Step 15300, Train Loss: 3.1135, Val Loss: 3.0749, LR: 0.000059, Tokens/s: 187,641
Step 15400, Train Loss: 3.1204, Val Loss: 3.0728, LR: 0.000057, Tokens/s: 187,462
... (truncated) ...
Step 18900, Train Loss: 3.0411, Val Loss: 3.0494, LR: 0.000030, Tokens/s: 187,473
Model saved to model.pt
```

## File Overview
- `model.py`: Transformer model and rotary embedding implementation.
- `train.py`: Pretraining script with distributed training support.
- `sft_train.py`: Supervised fine-tuning (SFT) script for instruction-following data.
- `sft_data.py`: Dataset and prompt templates for SFT.
- `prepare.py`: Data preparation and tokenization for pretraining.
- `data.py`: Dataset and sampler utilities for pretraining.
- `chat_apollo.py`: Inference/chat interface for the trained model.
- `test.py`: Generation utilities and model testing.

## Usage
1. **Prepare Data:**
	- Use `prepare.py` to tokenize and preprocess your dataset.
2. **Pretrain Model:**
	- Run `train.py` for distributed pretraining.
3. **Supervised Fine-Tuning:**
	- Use `sft_train.py` with your instruction-following data (e.g., Alpaca format).
4. **Inference/Chat:**
	- Use `chat_apollo.py` to interact with the trained model.

## Example: Pretraining
```bash
torchrun --nproc_per_node=8 train.py
```

## Example: SFT
```bash
torchrun --nproc_per_node=8 sft_train.py
```

## Example: Chat
```bash
python chat_apollo.py
```

## Training Results
- Training and validation loss decrease steadily, indicating effective learning.
- Final losses (example):
  - Train Loss: ~3.04
  - Val Loss: ~3.05
- Model checkpoint saved as `model.pt`.

## Requirements
- Python 3.8+
- PyTorch
- tiktoken
- datasets
- tqdm

## License
MIT