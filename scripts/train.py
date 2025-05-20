# scripts/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

def train_model(model, train_data, train_labels, dev_data, dev_labels, epochs=10, batch_size=32, lr=1e-3):
    dataset = TensorDataset(train_data, train_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output, _ = model(batch_x)
            loss = criterion(output.view(-1, output.size(-1)), batch_y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        evaluate_model(model, dev_data, dev_labels)


def evaluate_model(model, data, labels):
    model.eval()
    with torch.no_grad():
        output, _ = model(data)
        preds = output.argmax(dim=-1).view(-1).cpu().numpy()
        true = labels.view(-1).cpu().numpy()
        acc = accuracy_score(true, preds)
        f1 = f1_score(true, preds, average='macro')
        print(f"Eval Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")