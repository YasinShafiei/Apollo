import os
import sys
import torch
import tiktoken

from model import Model

# model config (must match your trained model)
VOCAB_SIZE = 50257
EMBED_DIM = 1024
N_LAYERS = 20
N_HEADS = 16
N_KV_HEADS = 8
HIDDEN_DIM = 2048
MAX_SEQ_LEN = 1024

# device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# prompt template (must match sft_data.py)
PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Response:
"""

PROMPT_TEMPLATE_WITH_INPUT = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

# ANSI color codes for terminal UI
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    print()
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("  ┌─────────────────────────────────────────┐")
    print("  │                                         │")
    print("  │            ✦  A P O L L O  ✦            │")
    print("  │                                         │")
    print("  │         Your Local Language Model       │")
    print("  │                                         │")
    print("  └─────────────────────────────────────────┘")
    print(f"{Colors.RESET}")


def print_status(message, color=Colors.DIM):
    print(f"{color}  {message}{Colors.RESET}")


def print_help():
    print(f"\n{Colors.DIM}  Commands:{Colors.RESET}")
    print(f"{Colors.DIM}    • Type your message and press Enter{Colors.RESET}")
    print(f"{Colors.DIM}    • 'clear' - Clear the screen{Colors.RESET}")
    print(f"{Colors.DIM}    • 'temp <value>' - Set temperature (0.1-2.0){Colors.RESET}")
    print(f"{Colors.DIM}    • 'tokens <value>' - Set max tokens (1-1024){Colors.RESET}")
    print(f"{Colors.DIM}    • 'help' - Show this help{Colors.RESET}")
    print(f"{Colors.DIM}    • 'quit' or 'exit' - Exit the program{Colors.RESET}\n")


def format_prompt(instruction, input_text=None):
    if input_text:
        return PROMPT_TEMPLATE_WITH_INPUT.format(instruction=instruction, input=input_text)
    return PROMPT_TEMPLATE.format(instruction=instruction)


def generate(model, enc, prompt_ids, max_new_tokens=256, temperature=0.7, top_k=40, top_p=0.9):
    model.eval()
    ids = prompt_ids.clone()
    generated_tokens = 0
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids if ids.size(1) <= MAX_SEQ_LEN else ids[:, -MAX_SEQ_LEN:]
            
            logits, _ = model(context)
            logits = logits[:, -1, :]
            
            if temperature > 0:
                logits = logits / temperature
                
                # top-k filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # top-p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)
            
            if next_id.item() == enc.eot_token:
                break
            
            token_text = enc.decode([next_id.item()])
            print(token_text, end="", flush=True)
            generated_tokens += 1
            
            ids = torch.cat([ids, next_id], dim=1)
    
    print()
    return generated_tokens


def load_model(model_path="model_sft.pt"):
    print_status(f"Loading model on {device}...", Colors.YELLOW)
    
    model = Model(VOCAB_SIZE, EMBED_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, MAX_SEQ_LEN).to(device)
    
    if not os.path.exists(model_path):
        print(f"\n{Colors.RED}  Error: Model file '{model_path}' not found!{Colors.RESET}")
        print(f"{Colors.DIM}  Make sure model_sft.pt is in the current directory.{Colors.RESET}\n")
        sys.exit(1)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    param_count = sum(p.numel() for p in model.parameters())
    print_status(f"Model loaded: {param_count:,} parameters", Colors.GREEN)
    
    return model


def main():
    clear_screen()
    print_header()
    
    enc = tiktoken.get_encoding("gpt2")
    model = load_model()
    
    # default generation settings
    temperature = 0.7
    max_tokens = 256
    
    print_help()
    print(f"{Colors.DIM}  ─────────────────────────────────────────{Colors.RESET}\n")
    
    while True:
        try:
            print(f"{Colors.GREEN}{Colors.BOLD}  You ›{Colors.RESET} ", end="")
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n{Colors.CYAN}  Goodbye! 👋{Colors.RESET}\n")
            break
        
        if not user_input:
            continue
        
        # handle commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print(f"\n{Colors.CYAN}  Goodbye! 👋{Colors.RESET}\n")
            break
        
        if user_input.lower() == 'clear':
            clear_screen()
            print_header()
            print(f"{Colors.DIM}  ─────────────────────────────────────────{Colors.RESET}\n")
            continue
        
        if user_input.lower() == 'help':
            print_help()
            continue
        
        if user_input.lower().startswith('temp '):
            try:
                new_temp = float(user_input.split()[1])
                if 0.1 <= new_temp <= 2.0:
                    temperature = new_temp
                    print_status(f"Temperature set to {temperature}", Colors.GREEN)
                else:
                    print_status("Temperature must be between 0.1 and 2.0", Colors.RED)
            except (ValueError, IndexError):
                print_status("Usage: temp <value>", Colors.RED)
            continue
        
        if user_input.lower().startswith('tokens '):
            try:
                new_tokens = int(user_input.split()[1])
                if 1 <= new_tokens <= 1024:
                    max_tokens = new_tokens
                    print_status(f"Max tokens set to {max_tokens}", Colors.GREEN)
                else:
                    print_status("Max tokens must be between 1 and 1024", Colors.RED)
            except (ValueError, IndexError):
                print_status("Usage: tokens <value>", Colors.RED)
            continue
        
        # generate response
        prompt = format_prompt(user_input)
        prompt_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
        
        print(f"\n{Colors.CYAN}{Colors.BOLD}  Apollo ›{Colors.RESET} ", end="")
        
        try:
            tokens_generated = generate(
                model, enc, prompt_ids,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            print(f"{Colors.DIM}  [{tokens_generated} tokens]{Colors.RESET}\n")
        except Exception as e:
            print(f"\n{Colors.RED}  Error during generation: {e}{Colors.RESET}\n")


if __name__ == "__main__":
    main()