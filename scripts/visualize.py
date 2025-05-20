# scripts/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformer_model.model import TransformerModel
from utils.data_utils import build_vocab, index_string

def plot_attention(model, input_str, vocab, save=False, save_path="attention_layer{}.png"):
    model.eval()
    idx = index_string(input_str, vocab).to(next(model.parameters()).device)
    with torch.no_grad():
        _, attn_maps = model(idx)

    for layer_idx, attn in enumerate(attn_maps):
        if attn.dim() == 4:
            attn = attn.squeeze(0)[0]  # for 1 head

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(attn.cpu().numpy(),
                    xticklabels=list(input_str),
                    yticklabels=list(input_str),
                    cmap="viridis", ax=ax)
        ax.set_title(f"Layer {layer_idx + 1} Attention Map")

        if save:
            plt.savefig(save_path.format(layer_idx + 1))
        else:
            plt.show()
