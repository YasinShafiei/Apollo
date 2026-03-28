import os
import sys
import torch
import tiktoken

from model import Model
from config import ModelConfig
from sft_data import IM_START, IM_END

cfg = ModelConfig()
enc = tiktoken.get_encoding("cl100k_base")

# device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class Colors:
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
    print(f"\n{Colors.CYAN}{Colors.BOLD}")
    print("  ┌─────────────────────────────────────────┐")
    print("  │                                         │")
    print("  │            ✦  A P O L L O  ✦            │")
    print("  │                                         │")
    print("  │         Your Local Language Model       │")
    print("  │                                         │")
    print("  └─────────────────────────────────────────┘")
    print(Colors.RESET)


def print_status(msg, color=Colors.DIM):
    print(f"{color}  {msg}{Colors.RESET}")


def print_help():
    print(f"\n{Colors.DIM}  Commands:{Colors.RESET}")
    print(f"{Colors.DIM}    • Type your message and press Enter{Colors.RESET}")
    print(f"{Colors.DIM}    • 'clear'  - Clear screen & conversation{Colors.RESET}")
    print(f"{Colors.DIM}    • 'temp N' - Set temperature (0.0-2.0, default 0.6){Colors.RESET}")
    print(f"{Colors.DIM}    • 'tokens N' - Set max tokens (1-512, default 256){Colors.RESET}")
    print(f"{Colors.DIM}    • 'help'   - Show this help{Colors.RESET}")
    print(f"{Colors.DIM}    • 'quit'   - Exit{Colors.RESET}\n")


# ---------------------------------------------------------------------------
# Tokenization — matches sft_data.py _tokenize_conversation exactly
# ---------------------------------------------------------------------------

def tokenize_message(role, content):
    """<|im_start|>role\ncontent<|im_end|>"""
    role_toks = enc.encode(role + "\n", disallowed_special=())
    content_toks = enc.encode(content, disallowed_special=())
    return [IM_START] + role_toks + content_toks + [IM_END]


def build_prompt(messages, max_prompt_len):
    """
    Tokenize messages into ChatML and append the assistant generation prefix.
    Drops oldest turns (except the latest user message) if over max_prompt_len.
    """
    # assistant generation prefix: <|im_start|>assistant\n
    asst_prefix = [IM_START] + enc.encode("assistant\n", disallowed_special=())

    # tokenize each message
    tokenized = [tokenize_message(m["role"], m["content"]) for m in messages]

    # keep as many recent turns as fit in max_prompt_len
    budget = max_prompt_len - len(asst_prefix)
    selected = []
    for toks in reversed(tokenized):
        if budget - len(toks) < 0 and selected:
            break
        selected.append(toks)
        budget -= len(toks)
    selected.reverse()

    prompt = []
    for toks in selected:
        prompt.extend(toks)
    prompt.extend(asst_prefix)
    return prompt


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, prompt_tokens, max_new_tokens=256,
             temperature=0.6, top_k=40, top_p=0.9,
             repetition_penalty=1.2):
    model.eval()
    ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated = []

    for _ in range(max_new_tokens):
        ctx = ids[:, -cfg.max_seq_len:]
        logits, _ = model(ctx)
        logits = logits[:, -1, :]  # (1, vocab)

        # repetition penalty on already-generated tokens
        if repetition_penalty != 1.0 and generated:
            seen = torch.tensor(list(set(generated)), device=device)
            s = logits[0, seen]
            logits[0, seen] = torch.where(s > 0,
                                          s / repetition_penalty,
                                          s * repetition_penalty)

        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature

            # top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cum_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                logits[remove.scatter(1, sorted_idx, remove)] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        tok = next_id.item()

        # stop on ChatML boundaries or end-of-text
        if tok in (IM_END, IM_START, enc.eot_token):
            break

        text = enc.decode([tok])
        print(text, end="", flush=True)
        generated.append(tok)
        ids = torch.cat([ids, next_id], dim=1)

    print()
    return enc.decode(generated)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    # try root first, then models/ dir
    for path in ["apollo_sft.pt", "models/apollo_sft.pt"]:
        if os.path.exists(path):
            model_path = path
            break
    else:
        print(f"\n{Colors.RED}  Error: apollo_sft.pt not found!{Colors.RESET}")
        print(f"{Colors.DIM}  Place it in the project root or models/ directory.{Colors.RESET}\n")
        sys.exit(1)

    print_status(f"Loading {model_path} on {device}...", Colors.YELLOW)

    model = Model(
        cfg.vocab_size, cfg.embed_dim, cfg.n_layers, cfg.n_heads,
        cfg.n_kv_heads, cfg.hidden_dim, cfg.max_seq_len
    ).to(device)

    state = torch.load(model_path, map_location=device, weights_only=True)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)

    params = sum(p.numel() for p in model.parameters())
    print_status(f"Model loaded: {params:,} parameters ({device})", Colors.GREEN)
    return model


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------

def main():
    clear_screen()
    print_header()
    model = load_model()

    temperature = 0.6
    max_tokens = 256
    messages = []

    print_help()
    print(f"{Colors.DIM}  ─────────────────────────────────────────{Colors.RESET}\n")

    while True:
        try:
            print(f"{Colors.GREEN}{Colors.BOLD}  You ›{Colors.RESET} ", end="")
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n{Colors.CYAN}  Goodbye!{Colors.RESET}\n")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ('quit', 'exit', 'q'):
            print(f"\n{Colors.CYAN}  Goodbye!{Colors.RESET}\n")
            break

        if cmd == 'clear':
            clear_screen()
            print_header()
            messages.clear()
            print_status("Conversation cleared.", Colors.GREEN)
            print(f"{Colors.DIM}  ─────────────────────────────────────────{Colors.RESET}\n")
            continue

        if cmd == 'help':
            print_help()
            continue

        if cmd.startswith('temp '):
            try:
                val = float(cmd.split()[1])
                if 0.0 <= val <= 2.0:
                    temperature = val
                    print_status(f"Temperature set to {temperature}", Colors.GREEN)
                else:
                    print_status("Must be between 0.0 and 2.0", Colors.RED)
            except (ValueError, IndexError):
                print_status("Usage: temp <value>", Colors.RED)
            continue

        if cmd.startswith('tokens '):
            try:
                val = int(cmd.split()[1])
                if 1 <= val <= 512:
                    max_tokens = val
                    print_status(f"Max tokens set to {max_tokens}", Colors.GREEN)
                else:
                    print_status("Must be between 1 and 512", Colors.RED)
            except (ValueError, IndexError):
                print_status("Usage: tokens <value>", Colors.RED)
            continue

        # add user message and build prompt
        messages.append({"role": "user", "content": user_input})
        max_prompt_len = cfg.max_seq_len - max_tokens
        prompt_tokens = build_prompt(messages, max_prompt_len)

        print(f"\n{Colors.CYAN}{Colors.BOLD}  Apollo ›{Colors.RESET} ", end="")

        try:
            response = generate(
                model, prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            print(f"{Colors.DIM}  [{len(enc.encode(response, disallowed_special=()))} tokens]{Colors.RESET}\n")

            if response.strip():
                messages.append({"role": "assistant", "content": response})
            else:
                messages.pop()  # remove user msg if no response

        except Exception as e:
            print(f"\n{Colors.RED}  Error: {e}{Colors.RESET}\n")
            messages.pop()


if __name__ == "__main__":
    main()
