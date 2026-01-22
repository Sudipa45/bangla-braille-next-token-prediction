#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, json, argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


# -----------------------------
# 1) Braille -> Bangla fallback (lossy)
# -----------------------------
def build_inverse_map() -> Dict[str, str]:
    # MUST match (roughly) the mapping you used during dataset creation.
    # This is only a fallback if you don't have input_bn/target_bn columns.
    br2bn = {
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
        # unknown placeholder from earlier scripts
        "⣿":""
    }
    return br2bn

def braille_tokens_to_bangla_text(input_braille: str, br2bn: Dict[str,str]) -> str:
    # input_braille is "tok tok tok"
    toks = input_braille.split()
    return "".join(br2bn.get(t, "") for t in toks).strip()


# -----------------------------
# 2) Dataset
# -----------------------------
class BrailleXLMRDataset(Dataset):
    """
    Uses:
      - If CSV has input_bn, uses that (BEST)
      - else reconstructs Bangla from input_braille using inverse mapping (fallback)
    Labels: target_id (next Braille token id)
    """
    def __init__(self, csv_path: str, tokenizer, max_len: int = 96):
        self.rows = []
        self.tok = tokenizer
        self.max_len = max_len
        self.br2bn = build_inverse_map()

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            has_bn = ("input_bn" in fieldnames)

            for r in reader:
                if has_bn:
                    text = r["input_bn"]
                else:
                    text = braille_tokens_to_bangla_text(r["input_braille"], self.br2bn)

                y = int(r["target_id"])
                self.rows.append((text, y))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        text, y = self.rows[idx]
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(y, dtype=torch.long),
        }

def collate_pad(batch, pad_id: int):
    input_ids = [b["input_ids"] for b in batch]
    attn = [b["attention_mask"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])

    max_len = max(x.size(0) for x in input_ids)
    def pad1(x, val):
        if x.size(0) < max_len:
            return torch.cat([x, torch.full((max_len - x.size(0),), val, dtype=x.dtype)], dim=0)
        return x

    input_ids = torch.stack([pad1(x, pad_id) for x in input_ids])
    attn = torch.stack([pad1(x, 0) for x in attn])
    return input_ids, attn, labels


# -----------------------------
# 3) Model: XLM-R encoder + classification head
# -----------------------------
class XLMRNextToken(nn.Module):
    def __init__(self, backbone_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(backbone_name)
        hid = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hid, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [CLS]
        logits = self.fc(self.drop(cls))
        return logits


# -----------------------------
# 4) Metrics
# -----------------------------
@torch.no_grad()
def eval_topk(model, loader, device, topk=(1,3,5)):
    model.eval()
    correct = {k: 0 for k in topk}
    total = 0
    for input_ids, attn, y in loader:
        input_ids, attn, y = input_ids.to(device), attn.to(device), y.to(device)
        logits = model(input_ids, attn)
        total += y.size(0)
        for k in topk:
            preds = torch.topk(logits, k=k, dim=-1).indices
            correct[k] += (preds == y.unsqueeze(1)).any(dim=1).sum().item()
    return {k: correct[k]/total for k in topk}


# -----------------------------
# 5) Train loop (mixed precision)
# -----------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    train_ds = BrailleXLMRDataset(os.path.join(args.data_dir, "train.csv"), tok, max_len=args.max_len)
    val_ds   = BrailleXLMRDataset(os.path.join(args.data_dir, "val.csv"), tok, max_len=args.max_len)
    test_ds  = BrailleXLMRDataset(os.path.join(args.data_dir, "test.csv"), tok, max_len=args.max_len)

    # num labels from vocab.json
    with open(os.path.join(args.data_dir, "vocab.json"), "r", encoding="utf-8") as f:
        tok2id = json.load(f)
    num_labels = max(tok2id.values()) + 1

    collate = lambda b: collate_pad(b, pad_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = XLMRNextToken(args.model_name, num_labels=num_labels, dropout=args.dropout).to(device)

    # Freeze option (fast)
    if args.freeze_encoder:
        for p in model.enc.parameters():
            p.requires_grad = False

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=max(10, total_steps//20), num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device == "cuda")
    use_amp = (device == "cuda") and (args.fp16 or args.bf16)
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    crit = nn.CrossEntropyLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, "best.pt")
    best_val = -1.0

    for ep in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        n = 0

        for input_ids, attn, y in train_loader:
            input_ids, attn, y = input_ids.to(device), attn.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(input_ids, attn)
                    loss = crit(logits, y)
            else:
                logits = model(input_ids, attn)
                loss = crit(logits, y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            sched.step()
            total_loss += loss.item() * y.size(0)
            n += y.size(0)

        val_acc = eval_topk(model, val_loader, device, topk=(1,3,5))
        print(f"Epoch {ep:02d} | loss={total_loss/max(n,1):.4f} | val@1={val_acc[1]:.4f} val@3={val_acc[3]:.4f} val@5={val_acc[5]:.4f}")

        if val_acc[1] > best_val:
            best_val = val_acc[1]
            torch.save({"model": model.state_dict(), "epoch": ep, "val@1": best_val}, best_path)

    # test best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_acc = eval_topk(model, test_loader, device, topk=(1,3,5))
    print(f"\nBest: epoch={ckpt['epoch']} val@1={ckpt['val@1']:.4f}")
    print(f"TEST: top1={test_acc[1]:.4f} top3={test_acc[3]:.4f} top5={test_acc[5]:.4f}")
    print("Saved:", best_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="runs/xlmr_braille")
    ap.add_argument("--model_name", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=96)

    ap.add_argument("--freeze_encoder", action="store_true")

    # mixed precision
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    if args.fp16 and args.bf16:
        raise SystemExit("Choose only one: --fp16 OR --bf16")

    train(args)

if __name__ == "__main__":
    main()
