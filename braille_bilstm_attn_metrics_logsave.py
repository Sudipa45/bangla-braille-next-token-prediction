#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, csv, argparse, math, time, platform
from dataclasses import asdict
from typing import Dict, Tuple, List

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
    xpad = pad_sequence(xs, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    return xpad, lengths, ys


# -----------------------------
# Model: BiLSTM + Attention
# -----------------------------
class BiLSTMAttnNextToken(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.2, attn_dim: int = 256):
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

        self.dropout = nn.Dropout(dropout)

        # Attention: score_t = v^T tanh(W h_t)
        self.attn_W = nn.Linear(2 * hidden_dim, attn_dim, bias=True)
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)

        self.fc = nn.Linear(2 * hidden_dim, vocab_size)

    def forward(self, x, lengths):
        e = self.emb(x)  # [B, T, E]
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # [B, T, 2H]
        out = self.dropout(out)

        B, T, _ = out.shape
        mask = torch.arange(T, device=out.device).unsqueeze(0) < lengths.unsqueeze(1)  # [B, T]

        scores = self.attn_v(torch.tanh(self.attn_W(out))).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(~mask, -1e9)

        alpha = torch.softmax(scores, dim=1)  # [B, T]
        ctx = torch.bmm(alpha.unsqueeze(1), out).squeeze(1)  # [B, 2H]

        logits = self.fc(ctx)  # [B, V]
        return logits


# -----------------------------
# Loss
# -----------------------------
def make_criterion(label_smoothing: float):
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def eval_metrics(model, loader, device, topk=(1, 3, 5)) -> Dict[str, float]:
    model.eval()
    ce_nosmooth = nn.CrossEntropyLoss()

    correct = {k: 0 for k in topk}
    total = 0

    mrr_sum = 0.0
    rank_sum = 0.0
    loss_sum = 0.0

    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        logits = model(x, lengths)

        loss = ce_nosmooth(logits, y)
        loss_sum += loss.item() * y.size(0)
        total += y.size(0)

        for k in topk:
            preds = torch.topk(logits, k=k, dim=-1).indices
            correct[k] += (preds == y.unsqueeze(1)).any(dim=1).sum().item()

        # Rank of the true label among all labels: 1 + #greater logits
        true_logit = logits.gather(1, y.unsqueeze(1))  # [B,1]
        rank = 1 + (logits > true_logit).sum(dim=1)     # [B]
        rank = rank.to(torch.float32)

        rank_sum += rank.sum().item()
        mrr_sum += (1.0 / rank).sum().item()

    out = {}
    for k in topk:
        out[f"top@{k}"] = correct[k] / total
        out[f"ksr@{k}"] = out[f"top@{k}"]  # explicit

    out["mrr"] = mrr_sum / total
    out["coverage_error"] = rank_sum / total

    mean_ce = loss_sum / total
    out["mean_ce"] = mean_ce
    out["ppl"] = math.exp(mean_ce) if mean_ce < 50 else float("inf")
    return out


def fmt_metrics(m: Dict[str, float], topk=(1, 3, 5)) -> str:
    parts = []
    for k in topk:
        parts.append(f"top@{k}={m[f'top@{k}']:.4f}")
    parts.append(f"MRR={m['mrr']:.4f}")
    parts.append(f"Coverage={m['coverage_error']:.2f}")
    parts.append(f"PPL={m['ppl']:.3f}")
    for k in topk:
        parts.append(f"KSR@{k}={m[f'ksr@{k}']:.4f}")
    return " | ".join(parts)


# -----------------------------
# Logging helpers
# -----------------------------
class TeeLogger:
    """Print to console and append to a file."""
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def write(self, msg: str):
        print(msg, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_summary_text(model: nn.Module) -> str:
    lines = []
    lines.append("=== Model Architecture ===")
    lines.append(str(model))
    total, trainable = count_params(model)
    lines.append("")
    lines.append(f"Total parameters: {total:,}")
    lines.append(f"Trainable parameters: {trainable:,}")
    return "\n".join(lines)


def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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
    ap.add_argument("--attn_dim", type=int, default=256)

    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--out_dir", default="runs/bilstm_attn_braille")
    args = ap.parse_args()

    # Run setup
    os.makedirs(args.out_dir, exist_ok=True)
    log = TeeLogger(os.path.join(args.out_dir, "train.log"))

    # Paths
    vocab_path = os.path.join(args.data_dir, "vocab.json")
    train_csv  = os.path.join(args.data_dir, "train.csv")
    val_csv    = os.path.join(args.data_dir, "val.csv")
    test_csv   = os.path.join(args.data_dir, "test.csv")

    # Vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        tok2id = json.load(f)
    vocab_size = max(tok2id.values()) + 1

    # Device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_name = torch.cuda.get_device_name(0) if device == "cuda" else None

    # Data
    train_ds = BrailleNextTokenDataset(train_csv)
    val_ds   = BrailleNextTokenDataset(val_csv)
    test_ds  = BrailleNextTokenDataset(test_csv)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    # Model
    model = BiLSTMAttnNextToken(
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        attn_dim=args.attn_dim
    ).to(device)

    criterion = make_criterion(args.label_smoothing)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Save config + summary
    run_info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "cuda_device": cuda_name,
        "data_dir": args.data_dir,
        "out_dir": args.out_dir,
        "vocab_size": vocab_size,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "hyperparameters": vars(args),
    }
    total_params, trainable_params = count_params(model)
    run_info["total_params"] = total_params
    run_info["trainable_params"] = trainable_params

    save_json(os.path.join(args.out_dir, "run_config.json"), run_info)

    summary_txt = model_summary_text(model)
    with open(os.path.join(args.out_dir, "model_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_txt)

    # Print hyperparams + summary
    log.write("=== Run Configuration ===")
    log.write(json.dumps(run_info, indent=2, ensure_ascii=False))
    log.write("")
    log.write(summary_txt)
    log.write("")

    # CSV training log
    metrics_csv_path = os.path.join(args.out_dir, "epoch_log.csv")
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch", "train_loss",
            "val_top1", "val_top3", "val_top5",
            "val_mrr", "val_coverage", "val_ppl", "val_mean_ce"
        ])

    best_val = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")
    last_path = os.path.join(args.out_dir, "last.pt")

    topk = (1, 3, 5)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optim, criterion, device)
        val_m = eval_metrics(model, val_loader, device, topk=topk)
        dt = time.time() - t0

        log.write(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"VAL: {fmt_metrics(val_m, topk=topk)} | time={dt:.1f}s"
        )

        with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                epoch, f"{train_loss:.6f}",
                f"{val_m['top@1']:.6f}", f"{val_m['top@3']:.6f}", f"{val_m['top@5']:.6f}",
                f"{val_m['mrr']:.6f}", f"{val_m['coverage_error']:.6f}",
                f"{val_m['ppl']:.6f}", f"{val_m['mean_ce']:.6f}"
            ])

        # Save last checkpoint every epoch
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_metrics": val_m,
            "args": vars(args),
        }, last_path)

        # Save best checkpoint on val top@1
        if val_m["top@1"] > best_val:
            best_val = val_m["top@1"]
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_metrics": val_m,
                "best_val_top1": best_val,
                "args": vars(args),
            }, best_path)

    # Evaluate best on test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_m = eval_metrics(model, test_loader, device, topk=topk)

    log.write("")
    log.write(f"=== Best checkpoint ===")
    log.write(f"epoch={ckpt['epoch']} best_val_top1={ckpt.get('best_val_top1', ckpt['val_metrics']['top@1']):.4f}")
    log.write(f"TEST: {fmt_metrics(test_m, topk=topk)}")
    log.write(f"Saved best: {best_path}")
    log.write(f"Saved last: {last_path}")
    log.write(f"Saved config: {os.path.join(args.out_dir, 'run_config.json')}")
    log.write(f"Saved model summary: {os.path.join(args.out_dir, 'model_summary.txt')}")
    log.write(f"Saved epoch log: {metrics_csv_path}")

    # Save final metrics JSON
    final_metrics = {
        "best_epoch": int(ckpt["epoch"]),
        "best_val_metrics": ckpt["val_metrics"],
        "test_metrics": test_m,
    }
    save_json(os.path.join(args.out_dir, "final_metrics.json"), final_metrics)


if __name__ == "__main__":
    main()
