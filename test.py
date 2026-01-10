import torch
import tiktoken

from model import Model

VOCAB_SIZE = 50257
EMBED_DIM = 1024  
N_LAYERS = 20
N_HEADS = 16
N_KV_HEADS = 8
HIDDEN_DIM = 2048
MAX_SEQ_LEN = 1024


device = "cuda" if torch.cuda.is_available() else "cpu"

def generate(model, prompt_ids, max_new_tokens=1024, temperature=0.8, top_k=50):
    model.eval()
    ids = prompt_ids.clone()
    enc = tiktoken.get_encoding("gpt2")
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # crop to max sequence length if needed
            context = ids if ids.size(1) <= MAX_SEQ_LEN else ids[:, -MAX_SEQ_LEN:]
            
            # forward pass
            logits, _ = model(context)
            logits = logits[:, -1, :] / temperature
            
            # top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # sample from distribution
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            
            # print token immediately
            token_text = enc.decode([next_id.item()])
            print(token_text, end="", flush=True)
            
            # append to sequence
            ids = torch.cat([ids, next_id], dim=1)
    
    print()  # newline at the end
    return ids


def main():
    # load tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # load model
    model = Model(VOCAB_SIZE, EMBED_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, MAX_SEQ_LEN).to(device)
    model.load_state_dict(torch.load("model.pt", map_location=device, weights_only=True))
    print(f"model loaded on {device}")
    
    # encode prompt
    prompt = "The meaning of life is"
    prompt_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
    print(f"generating 1024 tokens...\n")
    
    # print prompt first, then stream generated tokens
    print(prompt, end="", flush=True)
    generate(model, prompt_ids, max_new_tokens=1024)


if __name__ == "__main__":
    main()