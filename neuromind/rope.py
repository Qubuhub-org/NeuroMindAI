import torch

def precompute_rope(max_seq_len, head_dim, theta=10000.0, device=None):
    if head_dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    cos = cos[:x.size(2)].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.size(2)].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x * cos.repeat_interleave(2, -1) + rotated * sin.repeat_interleave(2, -1)
