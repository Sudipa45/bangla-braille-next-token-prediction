#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, csv, argparse, random
from typing import List, Dict
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

from peft import LoraConfig, get_peft_model, TaskType


# ---------------------------
# Utils
# ---------------------------
def load_vocab(vocab_path: str) -> List[str]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        tok2id = json.load(f)
    tokens = sorted(tok2id.keys(), key=lambda t: tok2id[t])
    return tokens

def read_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def build_prompt(context: str, allowed_tokens: List[str]) -> List[Dict[str, str]]:
    allowed = " ".join(allowed_tokens)
    system = (
        "You are a Braille next-token predictor. "
        "Output exactly ONE Braille token from the allowed list. No extra text."
    )
    user = (
        f"Context: {context}\n"
        f"Allowed: {allowed}\n"
        "Predict the next token. Output ONE token only."
    )
    return [{"role":"system","content":system},{"role":"user","content":user}]


# ---------------------------
# Dataset for instruction tuning
# ---------------------------
class BrailleTuneDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, allowed_tokens):
        self.rows = rows
        self.tok = tokenizer
        self.allowed_tokens = allowed_tokens

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        context = row["input_braille"]
        target = row["target_braille"]

        msgs = build_prompt(context, self.allowed_tokens)
        prompt_text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + target

        enc_full = self.tok(full_text, add_special_tokens=False)
        enc_prompt = self.tok(prompt_text, add_special_tokens=False)

        input_ids = enc_full["input_ids"]
        attn = enc_full["attention_mask"]

        # mask prompt tokens so loss only on the answer
        labels = [-100] * len(enc_prompt["input_ids"]) + input_ids[len(enc_prompt["input_ids"]):]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------
# Optional: Zero-shot logprob scoring
# ---------------------------
@torch.no_grad()
def score_candidates(model, tokenizer, messages, candidates: List[str], device: str) -> Dict[str, float]:
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    scores = {}
    for c in candidates:
        c_ids = tokenizer.encode(c, add_special_tokens=False)
        if len(c_ids) == 0:
            scores[c] = -1e9
            continue

        cur_input_ids = inputs["input_ids"]
        total = 0.0
        for tid in c_ids:
            out = model(input_ids=cur_input_ids)
            lp = torch.log_softmax(out.logits[:, -1, :], dim=-1)[0, tid].item()
            total += lp
            cur_input_ids = torch.cat([cur_input_ids, torch.tensor([[tid]], device=device)], dim=1)
        scores[c] = total
    return scores

def topk_from_scores(scores: Dict[str,float], k: int) -> List[str]:
    return [t for t,_ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--mode", choices=["tune_qlora","eval_zeroshot","eval_tuned"], required=True)
    ap.add_argument("--out_dir", default="runs/braille_qlora")
    ap.add_argument("--max_eval", type=int, default=2000)
    ap.add_argument("--topk", type=int, default=5)

    # training
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # precision
    ap.add_argument("--bf16", action="store_true", help="Use bf16 mixed precision (recommended if supported)")
    ap.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision")

    args = ap.parse_args()

    vocab_path = os.path.join(args.data_dir, "vocab.json")
    train_path = os.path.join(args.data_dir, "train.csv")
    val_path   = os.path.join(args.data_dir, "val.csv")
    test_path  = os.path.join(args.data_dir, "test.csv")

    allowed_tokens = load_vocab(vocab_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------
    # QLoRA 4-bit config
    # ---------------------------
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.mode in ["eval_zeroshot", "eval_tuned"]:
        model_path = args.model_name if args.mode == "eval_zeroshot" else args.out_dir

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()

        rows = read_csv(test_path)[:args.max_eval]
        top1 = 0
        topk = 0

        for row in tqdm(rows, desc="Evaluating"):
            msgs = build_prompt(row["input_braille"], allowed_tokens)
            scores = score_candidates(model, tokenizer, msgs, allowed_tokens, device=device)
            pred1 = topk_from_scores(scores, 1)[0]
            predk = set(topk_from_scores(scores, args.topk))
            gold = row["target_braille"]
            top1 += int(pred1 == gold)
            topk += int(gold in predk)

        n = len(rows)
        print(f"Top-1 Acc: {top1/n:.4f}")
        print(f"Top-{args.topk} Acc: {topk/n:.4f}")
        return

    # ---------------------------
    # Tune with QLoRA
    # ---------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False  # important for training

    # LoRA adapters
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    train_rows = read_csv(train_path)
    val_rows   = read_csv(val_path)

    train_ds = BrailleTuneDataset(train_rows, tokenizer, allowed_tokens)
    val_ds   = BrailleTuneDataset(val_rows, tokenizer, allowed_tokens)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Mixed precision settings:
    use_bf16 = bool(args.bf16)
    use_fp16 = bool(args.fp16) if not use_bf16 else False

    targs = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,

        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,

        bf16=use_bf16,
        fp16=use_fp16,

        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    print("Saved tuned QLoRA model to:", args.out_dir)


if __name__ == "__main__":
    main()
