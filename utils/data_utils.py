import torch

def build_vocab():
    vocab = {ch: i for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz ")}
    return vocab

def index_string(s, vocab):
    return torch.LongTensor([[vocab[c] for c in s]])

def task1_labels(s):
    labels = []
    count = {}
    for c in s:
        labels.append(min(count.get(c, 0), 2))
        count[c] = count.get(c, 0) + 1
    return torch.LongTensor([labels])

def task2_labels(s):
    labels = []
    for i, c in enumerate(s):
        before = s[:i].count(c)
        after = s[i+1:].count(c)
        total = before + after
        labels.append(min(total, 2))
    return torch.LongTensor([labels])

def load_dataset(path, vocab, task_fn):
    input_tensors = []
    label_tensors = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().lower()
            if len(line) != 20:  
                print(f"Skipping line (length {len(line)}): {line}")
                continue
            input_tensors.append(index_string(line, vocab))
            label_tensors.append(task_fn(line))
    return torch.cat(input_tensors, dim=0), torch.cat(label_tensors, dim=0)