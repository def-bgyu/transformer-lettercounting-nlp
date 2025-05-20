# main.py
import torch
from transformer_model.model import TransformerModel
from scripts.train import train_model
from utils.data_utils import build_vocab, load_dataset, task1_labels
from scripts.visualize import plot_attention
from utils.data_utils import build_vocab

if __name__ == "__main__":
    # Config
    vocab = build_vocab()
    vocab_size = len(vocab)
    d_model = 32
    max_seq_len = 20
    num_layers = 1
    num_heads = 1
    d_ff = 64
    num_classes = 3

    # Load data
    train_X, train_Y = load_dataset("data/lettercounting-train.txt", vocab, task1_labels)
    dev_X, dev_Y = load_dataset("data/lettercounting-dev.txt", vocab, task1_labels)

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_X, train_Y = train_X.to(device), train_Y.to(device)
    dev_X, dev_Y = dev_X.to(device), dev_Y.to(device)

    # Model
    model = TransformerModel(vocab_size, d_model, max_seq_len, num_layers, num_heads, d_ff, num_classes)
    model.to(device)

    # Train
    train_model(model, train_X, train_Y, dev_X, dev_Y, epochs=10, batch_size=32, lr=1e-3)

vocab = build_vocab()
plot_attention(model, "ed by rank and file", vocab, save=True)