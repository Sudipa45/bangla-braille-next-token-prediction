#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, csv, argparse, math
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# -----------------------------
# Dataset
# -----------------------------
class BrailleNextTokenDataset(Dataset):
    def __init__(self, csv_path: str):
        self.rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                # input_ids is a space-separated list of ints
                x = [int(t) for t in row["input_ids"].split()]
                y = int(row["target_id"])
                self.rows.append((x, y))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate_batch(batch):
    xs, ys = zip(*batch)
    xs = [torch.tensor(x, dtype=torch.long) for x in xs]
    ys = torch.tensor(ys, dtype=torch.long)
    # All your sequences are length 30, but padding keeps it robust
    xpad = pad_sequence(xs, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    return xpad, lengths, ys


# -----------------------------
# BiLSTM Model
# -----------------------------
class BiLSTMNextToken(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Because bidirectional => 2*hidden_dim
        self.fc = nn.Linear(2 * hidden_dim, vocab_size)

    def forward(self, x, lengths):
        # x: [B, T]
        e = self.emb(x)  # [B, T, E]
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, c_n) = self.lstm(packed)

        # h_n shape: [num_layers*2, B, hidden_dim]
        # Take last layer’s forward and backward hidden states:
        # forward is -2, backward is -1
        h_f = h_n[-2]  # [B, H]
        h_b = h_n[-1]  # [B, H]
        h = torch.cat([h_f, h_b], dim=-1)  # [B, 2H]

        logits = self.fc(h)  # [B, V]
        return logits


# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def eval_topk(model, loader, device, topk=(1,3,5)):
    model.eval()
    correct = {k: 0 for k in topk}
    total = 0

    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        logits = model(x, lengths)
        total += y.size(0)

        for k in topk:
            preds = torch.topk(logits, k=k, dim=-1).indices  # [B, k]
            correct[k] += (preds == y.unsqueeze(1)).any(dim=1).sum().item()

    return {k: correct[k]/total for k in topk}


# -----------------------------
# Train
# -----------------------------
def train_one_epoch(model, loader, optim, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        optim.zero_grad(set_to_none=True)
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    return total_loss / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--out_dir", default="runs/bilstm_braille")
    args = ap.parse_args()

    vocab_path = os.path.join(args.data_dir, "vocab.json")
    train_csv = os.path.join(args.data_dir, "train.csv")
    val_csv = os.path.join(args.data_dir, "val.csv")
    test_csv = os.path.join(args.data_dir, "test.csv")

    # vocab.json is token->id mapping
    with open(vocab_path, "r", encoding="utf-8") as f:
        tok2id = json.load(f)
    vocab_size = max(tok2id.values()) + 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = BrailleNextTokenDataset(train_csv)
    val_ds   = BrailleNextTokenDataset(val_csv)
    test_ds  = BrailleNextTokenDataset(test_csv)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = BiLSTMNextToken(
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = 0.0
    best_path = os.path.join(args.out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optim, criterion, device)
        val_acc = eval_topk(model, val_loader, device, topk=(1,3,5))
        print(f"Epoch {epoch:02d} | loss={loss:.4f} | val@1={val_acc[1]:.4f} val@3={val_acc[3]:.4f} val@5={val_acc[5]:.4f}")

        if val_acc[1] > best_val:
            best_val = val_acc[1]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val@1": best_val}, best_path)

    # Test best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_acc = eval_topk(model, test_loader, device, topk=(1,3,5))
    print(f"\nBest checkpoint: epoch={ckpt['epoch']} val@1={ckpt['val@1']:.4f}")
    print(f"TEST: top1={test_acc[1]:.4f} top3={test_acc[3]:.4f} top5={test_acc[5]:.4f}")
    print("Saved:", best_path)


if __name__ == "__main__":
    main()
