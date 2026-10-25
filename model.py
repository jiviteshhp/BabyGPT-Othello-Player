"""The notebook's own PyTorch GPT (cells 11 and 25), with unchanged weights."""

import torch
from torch import nn
from torch.nn import functional as F

VOCAB_SIZE = 66
CONTEXT_LENGTH = 60
EMBEDDING_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS):
        super().__init__()
        if embedding_dim % num_heads:
            raise ValueError("Embedding dimension must be divisible by head count")
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        batch, length, width = x.shape
        shape = (batch, length, self.num_heads, self.head_size)
        q = self.query(x).reshape(shape).transpose(1, 2)
        k = self.key(x).reshape(shape).transpose(1, 2)
        v = self.value(x).reshape(shape).transpose(1, 2)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        output = output.transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.projection(output)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class BabyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.position_embedding = nn.Embedding(CONTEXT_LENGTH, EMBEDDING_DIM)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(NUM_LAYERS)])
        self.norm = nn.LayerNorm(EMBEDDING_DIM)
        self.lm_head = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE)

    def forward(self, tokens):
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= CONTEXT_LENGTH:
            raise ValueError("Expected tokens with shape (batch, length), length 1..60")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.blocks(x)
        return self.lm_head(self.norm(x))


def load_model(checkpoint_path, device="cpu"):
    """Load a notebook training checkpoint or a plain state_dict, strictly on CPU first."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", checkpoint)
    model = BabyGPT()
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()
