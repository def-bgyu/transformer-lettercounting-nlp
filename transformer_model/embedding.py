# transformer_model/embedding.py
import torch.nn as nn
import torch

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        word_emb = self.word_embedding(x)
        pos_emb = self.positional_embedding(positions)
        return word_emb + pos_emb
