import math

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.checkpoint import checkpoint

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, base=10_000):
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base 

        # precompute frequences
        indices = torch.arange(0, self.dim, 2)
        inv_freqs = 1.0 / (base ** (indices / dim))
        self.register_buffer("inv_freqs", inv_freqs, persistent=False)

        t = torch.arange(0, max_seq_len)
        freqs = torch.einsum("i,j->ij", t, inv_freqs)

        # create cos and sin tables`
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x):
        B, H, T, C = x.shape

        # reshape x to (B * H, T, C)
        x = x.reshape(B*H, T, C)

        # break x 
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        # apply rotation 
        rot_even = x_even * self.cos[:T, :] - x_odd * self.sin[:T, :]
        rot_odd  = x_even * self.sin[:T, :] + x_odd * self.cos[:T, :]

        rotated = torch.stack((rot_even, rot_odd), dim=-1).flatten(-2, -1)

        # reshape back to (B, H, T, C)
        rotated = rotated.view(B, H, T, C)

        return rotated
    

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads, max_seq_len, dropout=0.0):
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.dropout = dropout

        self.head_dim = embed_dim // n_heads
        self.n_reps = n_heads // n_kv_heads 

        # define q, k, v projection matrices
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.w_v = nn.Linear(embed_dim, self.head_dim * self.n_kv_heads, bias=False)

        # out projection 
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, rope=None):
        B, T, C = x.shape 

        # calculate q k v 
        q = self.w_q(x) # (B, T, C)
        k = self.w_k(x)
        v = self.w_v(x)

        # split into heads (B, T, n_heads, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(-2, -3)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(-2, -3)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(-2, -3)

        # applying rope 
        if rope is not None:
            q = rope(q)
            k = rope(k)

        # repeating k and v
        k = k.unsqueeze(2).expand(B, self.n_kv_heads, self.n_reps, T, self.head_dim)
        k = k.reshape(B, self.n_kv_heads * self.n_reps, T, self.head_dim)

        v = v.unsqueeze(2).expand(B, self.n_kv_heads, self.n_reps, T, self.head_dim)
        v = v.reshape(B, self.n_kv_heads * self.n_reps, T, self.head_dim)

        # calculate attention weights
        # Note: dropout is only applied during training
        dropout_p = self.dropout if self.training else 0.0
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)

        # put everything back together
        out = att.transpose(-2, -3).contiguous().view(B, T, C)
        out = self.out_proj(out)

        return out
    
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        return (x / rms) * self.weight
    
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, dropout=0.2):
        super().__init__()
        
        hidden_dim = hidden_dim or int((8 * embed_dim) / 3)
        hidden_dim = (hidden_dim // 64) * 64

        # define gates and parameters
        self.w_gate = nn.Linear(embed_dim, hidden_dim)
        self.w_v = nn.Linear(embed_dim, hidden_dim)
        self.w_out = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        gate = F.silu(self.w_gate(x))
        v    = self.w_v(x)
        out  = self.w_out(gate * v)

        return self.dropout(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads, max_seq_len, hidden_dim=None, dropout=0.0):
        super().__init__()

        # define norm layers
        self.rms1 = RMSNorm(embed_dim)
        self.rms2 = RMSNorm(embed_dim)

        # define attention
        self.gqa = GroupedQueryAttention(embed_dim=embed_dim, n_heads=n_heads, n_kv_heads=n_kv_heads, max_seq_len=max_seq_len, dropout=dropout)

        # ffn
        self.ffn = FeedForward(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x, rope=None):
        x = x + self.gqa(self.rms1(x), rope=rope)
        x = x + self.ffn(self.rms2(x))

        return x
    
class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layers, n_heads, n_kv_heads, hidden_dim, max_seq_len, dropout=0.0):
        super().__init__()
        # token embedding 
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)

        # define RoPE
        self.rope = RotaryEmbedding(embed_dim//n_heads, max_seq_len)

        # transformer blocks
        self.layers = nn.ModuleList(TransformerBlock(embed_dim,
                                                     n_heads,
                                                     n_kv_heads,
                                                     max_seq_len,
                                                     hidden_dim=hidden_dim,
                                                     dropout=dropout) for _ in range(n_layers))

        # define normalization    
        self.rms_norm = RMSNorm(embed_dim)
        # final head 
        self.f_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):

        # embedding
        out = self.tok_emb(idx)
        
        # transformer blocks (with gradient checkpointing)
        for layer in self.layers:
            out = checkpoint(layer, out, self.rope, use_reentrant=False)

        # logits 
        out = self.rms_norm(out)

        # loss calculation
        loss = None
        if targets is not None:
            # chunked cross-entropy: never materialize full logits
            loss = self._chunked_cross_entropy(out, targets, chunk_size=512)
            return None, loss

        logits = self.f_head(out) # (B, T, vocab_size)
        
        return logits, loss

    def _chunked_cross_entropy(self, hidden, targets, chunk_size=512):
        B, T, C = hidden.shape
        total_loss = 0.0
        num_chunks = (T + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            s = i * chunk_size
            e = min(s + chunk_size, T)
            logits_chunk = self.f_head(hidden[:, s:e, :]).float()
            targets_chunk = targets[:, s:e]
            total_loss += F.cross_entropy(
                logits_chunk.reshape(-1, logits_chunk.size(-1)),
                targets_chunk.reshape(-1),
                reduction='sum'
            )
        return total_loss / (B * T)

if __name__ == "__main__":
    # Simple test parameters
    vocab_size = 100288
    embed_dim = 768  # Must be divisible by n_heads
    n_layers = 16
    n_heads = 12
    n_kv_heads = 4
    hidden_dim = 3072
    max_seq_len = 512
    
    # Create model
    model = Model(vocab_size, embed_dim, n_layers, n_heads, n_kv_heads, hidden_dim, max_seq_len)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Dummy input: (batch_size, seq_len)
    x = torch.randint(0, vocab_size, (4, max_seq_len))
    targets = torch.randint(0, vocab_size, (4, max_seq_len))
    
    # Forward pass without targets (just logits)
    logits, _ = model(x)
    print(f"Logits shape: {logits.shape}")
    
    # Forward pass with targets (loss + logits)
    _, loss = model(x, targets)
    print(f"Loss: {loss.item():.4f}")