import torch.nn as nn
import torch
from transformer_model.attention import MultiHeadAttention

# transformer_model/encoder.py

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # Attention block with residual connection
        attn_output, attn_weights = self.self_attn(x)
        x = x + attn_output  # no LayerNorm as per assignment spec

        # Feedforward block with residual connection
        ff_output = self.feed_forward(x)
        x = x + ff_output  # no LayerNorm as per assignment spec

        return x, attn_weights
