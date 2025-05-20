# transformer_model/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_model.embedding import TransformerEmbedding
from transformer_model.encoder import TransformerEncoderLayer

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, num_layers, num_heads, d_ff, num_classes):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_seq_len)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, x):
        attn_maps = []
        x = self.embedding(x)
        for layer in self.encoder_layers:
            x, attn = layer(x)
            attn_maps.append(attn)
        logits = self.output_layer(x)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, attn_maps