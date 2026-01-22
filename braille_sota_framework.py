#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, json, argparse
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer, AutoModel


# -----------------------------
# 0) Utilities
# -----------------------------
def load_tok2id(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)

def has_column(csv_path: str, col: str) -> bool:
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return (col in (r.fieldnames or []))


# -----------------------------
# 1) Braille -> Bangla fallback (lossy)
# -----------------------------
def build_inverse_map() -> Dict[str, str]:
    # Rough inverse of the simplified BN->BR mapping used during your dataset creation.
    # Best practice: regenerate dataset with input_bn for full fidelity.
    return {
        "␣": " ",
        "⠁":"অ","⠜":"আ","⠊":"ই","⠔":"ঈ","⠥":"উ","⠳":"ঊ","⠑":"এ","⠌":"ঐ","⠕":"ও","⠪":"ঔ",
        "⠅":"ক","⠭":"খ","⠛":"গ","⠣":"ঘ","⠬":"ঙ",
        "⠉":"চ","⠡":"ছ","⠚":"জ","⠵":"ঝ","⠻":"ঞ",
        "⠾":"ট","⠺":"ঠ","⠫":"ড","⠿":"ঢ","⠼":"ণ",
        "⠞":"ত","⠹":"থ","⠙":"দ","⠮":"ধ","⠝":"ন",
        "⠏":"প","⠖":"ফ","⠃":"ব","⠧":"ভ","⠍":"ম",
        "⠽":"য","⠗":"র","⠇":"ল","⠩":"শ","⠯":"ষ","⠎":"স","⠓":"হ",
        "⠲":"।","⠂":",","⠦":"?","⠤":"-",
        "⠈":"্","⠰":"ং","⠠":"ঃ","⠄":"ঁ",
        "⣿":""
    }

def braille_to_bn_text(input_braille: str) -> str:
    br2bn = build_inverse_map()
    toks = input_braille.split()
    return "".join(br2bn.get(t, "") for t in toks).strip()


# -----------------------------
# 2) Dataset (supports 3 modes)
#   - mode=bilstm: uses input_ids
#   - mode=xlmr: uses input_bn if exists else reconstruct
#   - mode=hybrid: both
# -----------------------------
class BrailleDataset(Dataset):
    def __init__(self, csv_path: str, mode: str, xlmr_tok=None, xlmr_max_len: int = 96):
        self.mode = mode
        self.rows = []
        self.xlmr_tok = xlmr_tok
        self.xlmr_max_len = xlmr_max_len

        self.has_bn = has_column(csv_path, "input_bn")

        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                x_ids = [int(t) for t in row["input_ids"].split()]
                y = int(row["target_id"])
                if mode in ("xlmr", "hybrid"):
                    if self.has_bn:
                        bn = row["input_bn"]
                    else:
                        bn = braille_to_bn_text(row["input_braille"])
                else:
                    bn = None
                self.rows.append((x_ids, bn, y))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        x_ids, bn, y = self.rows[idx]
        out = {"y": torch.tensor(y, dtype=torch.long)}

        if self.mode in ("bilstm", "hybrid"):
            out["x_ids"] = torch.tensor(x_ids, dtype=torch.long)
            out["len"] = torch.tensor(len(x_ids), dtype=torch.long)

        if self.mode in ("xlmr", "hybrid"):
            enc = self.xlmr_tok(
                bn,
                truncation=True,
                max_length=self.xlmr_max_len,
                padding=False,
                return_tensors=None
            )
            out["xlmr_input_ids"] = torch.tensor(enc["input_ids"], dtype=torch.long)
            out["xlmr_attn"] = torch.tensor(enc["attention_mask"], dtype=torch.long)

        return out


def collate_fn(mode: str, pad_id_x: int, pad_id_xlmr: int):
    def _collate(batch):
        ys = torch.stack([b["y"] for b in batch])

        out = {"y": ys}

        if mode in ("bilstm", "hybrid"):
            xs = [b["x_ids"] for b in batch]
            lens = torch.stack([b["len"] for b in batch])
            xpad = pad_sequence(xs, batch_first=True, padding_value=pad_id_x)
            out["x_ids"] = xpad
            out["len"] = lens

        if mode in ("xlmr", "hybrid"):
            ids = [b["xlmr_input_ids"] for b in batch]
            att = [b["xlmr_attn"] for b in batch]
            maxlen = max(t.size(0) for t in ids)

            def pad1(t, val):
                if t.size(0) < maxlen:
                    return torch.cat([t, torch.full((maxlen - t.size(0),), val, dtype=t.dtype)], dim=0)
                return t

            out["xlmr_input_ids"] = torch.stack([pad1(t, pad_id_xlmr) for t in ids])
            out["xlmr_attn"] = torch.stack([pad1(t, 0) for t in att])

        return out
    return _collate


# -----------------------------
# 3) Attention pooling over BiLSTM outputs
# -----------------------------
class AttnPool(nn.Module):
    """
    Computes attention weights over time:
      a_t = softmax(v^T tanh(W h_t))
      h = sum_t a_t h_t
    """
    def __init__(self, dim: int, attn_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, H: torch.Tensor, mask: torch.Tensor):
        # H: [B,T,D], mask: [B,T] (1 for valid)
        scores = self.v(torch.tanh(self.proj(H))).squeeze(-1)  # [B,T]
        scores = scores.masked_fill(mask == 0, -1e9)
        alpha = torch.softmax(scores, dim=-1)  # [B,T]
        pooled = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)  # [B,D]
        return pooled


# -----------------------------
# 4) Models
#   A) BiLSTM + attention
#   B) XLM-R + BiLSTM head (BiLSTM over XLM-R token embeddings)
#   C) Hybrid fusion: concat(BiLSTM-attn(Braille), BiLSTM-attn(XLM-R)) -> classifier
# -----------------------------
class BiLSTMAttn(nn.Module):
    def __init__(self, vocab_size: int, emb_dim=128, hidden_dim=256, layers=2, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers=layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.pool = AttnPool(dim=2*hidden_dim, attn_dim=hidden_dim)
        self.fc = nn.Linear(2*hidden_dim, vocab_size)

    def forward(self, x_ids, lengths):
        B, T = x_ids.size()
        e = self.emb(x_ids)  # [B,T,E]
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)  # [B,T,2H]
        mask = (torch.arange(T, device=x_ids.device).unsqueeze(0) < lengths.unsqueeze(1)).long()
        h = self.pool(H, mask)  # [B,2H]
        return self.fc(h)


class XLMR_BiLSTMAttn(nn.Module):
    def __init__(self, backbone: str, num_labels: int, hidden_dim=256, layers=1, dropout=0.2, freeze_backbone=False):
        super().__init__()
        self.xlmr = AutoModel.from_pretrained(backbone)
        if freeze_backbone:
            for p in self.xlmr.parameters():
                p.requires_grad = False

        d = self.xlmr.config.hidden_size
        self.lstm = nn.LSTM(
            d, hidden_dim, num_layers=layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.pool = AttnPool(dim=2*hidden_dim, attn_dim=hidden_dim)
        self.fc = nn.Linear(2*hidden_dim, num_labels)

    def forward(self, input_ids, attn):
        out = self.xlmr(input_ids=input_ids, attention_mask=attn)
        H0 = out.last_hidden_state  # [B,T,d]
        B, T, _ = H0.size()

        lengths = attn.sum(dim=1)  # [B]
        packed = nn.utils.rnn.pack_padded_sequence(H0, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)

        mask = attn.long()
        h = self.pool(H, mask)
        return self.fc(h)


class HybridFusion(nn.Module):
    def __init__(self, vocab_size: int, backbone: str, hidden_dim=256, dropout=0.2, freeze_backbone=False):
        super().__init__()
        # Braille branch
        self.braille_branch = BiLSTMAttn(vocab_size, emb_dim=128, hidden_dim=hidden_dim, layers=2, dropout=dropout)

        # XLM-R branch (outputs logits; we want features, so we reuse internals)
        self.xlmr = AutoModel.from_pretrained(backbone)
        if freeze_backbone:
            for p in self.xlmr.parameters():
                p.requires_grad = False
        d = self.xlmr.config.hidden_size
        self.xlmr_lstm = nn.LSTM(d, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.xlmr_pool = AttnPool(dim=2*hidden_dim, attn_dim=hidden_dim)

        # fusion classifier
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2*hidden_dim + vocab_size, vocab_size)  # simple fusion: [xlmr_feat | braille_logits]

    def forward(self, x_ids, lengths, xlmr_input_ids, xlmr_attn):
        # braille logits as a strong discriminative signal
        braille_logits = self.braille_branch(x_ids, lengths)  # [B,V]

        # xlmr pooled feature
        out = self.xlmr(input_ids=xlmr_input_ids, attention_mask=xlmr_attn)
        H0 = out.last_hidden_state  # [B,T,d]
        B, T, _ = H0.size()
        xlmr_len = xlmr_attn.sum(dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(H0, xlmr_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.xlmr_lstm(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        xlmr_feat = self.xlmr_pool(H, xlmr_attn.long())  # [B,2H]

        feat = torch.cat([xlmr_feat, braille_logits], dim=-1)
        feat = self.drop(feat)
        return self.fc(feat)


# -----------------------------
# 5) Label Smoothing
# -----------------------------
class LabelSmoothingCE(nn.Module):
    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        # logits: [B,V], target: [B]
        V = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)  # [B]
        smooth = -log_probs.mean(dim=-1)  # [B]
        return ((1 - self.eps) * nll + self.eps * smooth).mean()


# -----------------------------
# 6) Metrics (TopK + MRR)
# -----------------------------
@torch.no_grad()
def eval_metrics(model, loader, device, topk=(1,3,5)):
    model.eval()
    correct = {k: 0 for k in topk}
    total = 0
    mrr_sum = 0.0

    for batch in loader:
        y = batch["y"].to(device)

        if "x_ids" in batch and "xlmr_input_ids" in batch:
            logits = model(
                batch["x_ids"].to(device),
                batch["len"].to(device),
                batch["xlmr_input_ids"].to(device),
                batch["xlmr_attn"].to(device)
            )
        elif "x_ids" in batch:
            logits = model(batch["x_ids"].to(device), batch["len"].to(device))
        else:
            logits = model(batch["xlmr_input_ids"].to(device), batch["xlmr_attn"].to(device))

        total += y.size(0)

        # Top-k
        for k in topk:
            preds = torch.topk(logits, k=k, dim=-1).indices
            correct[k] += (preds == y.unsqueeze(1)).any(dim=1).sum().item()

        # MRR
        sorted_idx = torch.argsort(logits, dim=-1, descending=True)
        ranks = (sorted_idx == y.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1  # 1-based rank
        mrr_sum += (1.0 / ranks.float()).sum().item()

    out = {f"top{k}": correct[k]/total for k in topk}
    out["mrr"] = mrr_sum / total
    return out


# -----------------------------
# 7) Train loop
# -----------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    vocab_path = os.path.join(args.data_dir, "vocab.json")
    tok2id = load_tok2id(vocab_path)
    vocab_size = max(tok2id.values()) + 1

    # XLM-R tokenizer if needed
    xlmr_tok = None
    pad_id_xlmr = 1
    if args.mode in ("xlmr", "hybrid"):
        xlmr_tok = AutoTokenizer.from_pretrained(args.xlmr_name, use_fast=True)
        pad_id_xlmr = xlmr_tok.pad_token_id if xlmr_tok.pad_token_id is not None else xlmr_tok.eos_token_id

    train_ds = BrailleDataset(os.path.join(args.data_dir, "train.csv"), args.mode, xlmr_tok, args.xlmr_max_len)
    val_ds   = BrailleDataset(os.path.join(args.data_dir, "val.csv"),   args.mode, xlmr_tok, args.xlmr_max_len)
    test_ds  = BrailleDataset(os.path.join(args.data_dir, "test.csv"),  args.mode, xlmr_tok, args.xlmr_max_len)

    collate = collate_fn(args.mode, pad_id_x=0, pad_id_xlmr=pad_id_xlmr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Choose model
    if args.mode == "bilstm":
        model = BiLSTMAttn(vocab_size, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim,
                           layers=args.layers, dropout=args.dropout)
    elif args.mode == "xlmr":
        model = XLMR_BiLSTMAttn(args.xlmr_name, num_labels=vocab_size,
                                hidden_dim=args.hidden_dim, layers=1,
                                dropout=args.dropout, freeze_backbone=args.freeze_xlmr)
    else:
        model = HybridFusion(vocab_size, backbone=args.xlmr_name,
                             hidden_dim=args.hidden_dim, dropout=args.dropout,
                             freeze_backbone=args.freeze_xlmr)

    model = model.to(device)

    # Loss
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCE(eps=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0

        for batch in train_loader:
            y = batch["y"].to(device)
            optim.zero_grad(set_to_none=True)

            if args.mode == "hybrid":
                logits = model(
                    batch["x_ids"].to(device),
                    batch["len"].to(device),
                    batch["xlmr_input_ids"].to(device),
                    batch["xlmr_attn"].to(device)
                )
            elif args.mode == "bilstm":
                logits = model(batch["x_ids"].to(device), batch["len"].to(device))
            else:
                logits = model(batch["xlmr_input_ids"].to(device), batch["xlmr_attn"].to(device))

            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_loss += loss.item() * y.size(0)
            n += y.size(0)

        val = eval_metrics(model, val_loader, device)
        print(f"Epoch {ep:02d} | loss={total_loss/max(n,1):.4f} | "
              f"val top1={val['top1']:.4f} top3={val['top3']:.4f} top5={val['top5']:.4f} mrr={val['mrr']:.4f}")

        if val["top1"] > best_val:
            best_val = val["top1"]
            torch.save({"model": model.state_dict(), "epoch": ep, "val": val}, best_path)

    # Test best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test = eval_metrics(model, test_loader, device)
    print(f"\nBest checkpoint: epoch={ckpt['epoch']} val_top1={ckpt['val']['top1']:.4f}")
    print(f"TEST: top1={test['top1']:.4f} top3={test['top3']:.4f} top5={test['top5']:.4f} mrr={test['mrr']:.4f}")
    print("Saved:", best_path)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="runs/braille_sota")

    ap.add_argument("--mode", choices=["bilstm", "xlmr", "hybrid"], default="bilstm",
                    help="bilstm: Braille token BiLSTM; xlmr: XLM-R + BiLSTM; hybrid: fuse both")

    # BiLSTM
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)

    # XLM-R
    ap.add_argument("--xlmr_name", default="xlm-roberta-base")
    ap.add_argument("--xlmr_max_len", type=int, default=96)
    ap.add_argument("--freeze_xlmr", action="store_true")

    # training
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)

    # label smoothing
    ap.add_argument("--label_smoothing", type=float, default=0.1)

    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
