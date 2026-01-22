#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, csv, argparse, math
from typing import List, Dict
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

# Optional LoRA (recommended)
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_OK = True
except Exception:
    PEFT_OK = False


# ---------------------------
# Utils
# ---------------------------
def load_vocab(vocab_path: str) -> List[str]:
    # Your vocab.json in your folder is token->id (in your earlier script)
    with open(vocab_path, "r", encoding="utf-8") as f:
        tok2id = json.load(f)
    # sort by id
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
    system = "You are a Braille next-token predictor. Output exactly ONE Braille token from the allowed list. No extra text."
    user = f"Context: {context}\nAllowed: {allowed}\nPredict the next token. Output ONE token only."
    return [{"role":"system","content":system},{"role":"user","content":user}]


# ---------------------------
# Zero-shot: logprob scoring (classification)
# ---------------------------
@torch.no_grad()
def score_candidates(model, tokenizer, messages, candidates: List[str], device: str) -> Dict[str, float]:
    """
    Returns log p(candidate | prompt) for each candidate.
    We score the candidate as the log-prob of generating its tokenization next.
    """
    # Chat template → single string
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    # Get next-token distribution by running model once
    out = model(**inputs)
    logits = out.logits[:, -1, :]  # next token logits
    logprobs = torch.log_softmax(logits, dim=-1).squeeze(0)

    scores = {}
    for c in candidates:
        # candidate might tokenize into multiple tokens; approximate by summing sequentially
        # best practice: score multi-token candidates by autoregressive rollout
        # Here: rollout for correctness
        c_ids = tokenizer.encode(c, add_special_tokens=False)
        if len(c_ids) == 0:
            scores[c] = -1e9
            continue

        # autoregressive scoring
        cur_input_ids = inputs["input_ids"]
        total = 0.0
        for tid in c_ids:
            out2 = model(input_ids=cur_input_ids)
            lp = torch.log_softmax(out2.logits[:, -1, :], dim=-1)[0, tid].item()
            total += lp
            cur_input_ids = torch.cat([cur_input_ids, torch.tensor([[tid]], device=device)], dim=1)
        scores[c] = total

    return scores

def topk_from_scores(scores: Dict[str,float], k: int) -> List[str]:
    return [t for t,_ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


# ---------------------------
# Instruction-tuning dataset
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
        prompt_text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        # We want the model to generate ONLY target token.
        full_text = prompt_text + target

        enc = self.tok(full_text, return_tensors=None)
        # labels = input_ids, but mask prompt part so loss only on answer
        prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        input_ids = enc["input_ids"]
        labels = [-100]*len(prompt_ids) + input_ids[len(prompt_ids):]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--mode", choices=["zeroshot_eval","tune_lora","tuned_eval"], required=True)
    ap.add_argument("--out_dir", default="runs/braille_llm")
    ap.add_argument("--max_eval", type=int, default=2000)
    ap.add_argument("--topk", type=int, default=5)

    # tuning params
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    vocab_path = os.path.join(args.data_dir, "vocab.json")
    test_path  = os.path.join(args.data_dir, "test.csv")
    train_path = os.path.join(args.data_dir, "train.csv")
    val_path   = os.path.join(args.data_dir, "val.csv")

    allowed_tokens = load_vocab(vocab_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # IMPORTANT: ensure pad token exists for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.mode in ["zeroshot_eval", "tuned_eval"]:
        model_path = args.model_name if args.mode == "zeroshot_eval" else args.out_dir
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device=="cuda" else torch.float32,
            device_map="auto" if device=="cuda" else None
        ).to(device)
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

    if args.mode == "tune_lora":
        if not PEFT_OK:
            raise RuntimeError("peft not installed. Run: pip install peft")

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if device=="cuda" else torch.float32,
            device_map="auto" if device=="cuda" else None
        )
        model.config.use_cache = False

        # LoRA config (target modules depend on model; these work for many decoder-only LMs)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        )
        model = get_peft_model(model, lora_cfg)

        train_rows = read_csv(train_path)
        val_rows   = read_csv(val_path)

        train_ds = BrailleTuneDataset(train_rows, tokenizer, allowed_tokens)
        val_ds   = BrailleTuneDataset(val_rows, tokenizer, allowed_tokens)

        # pad to longest in batch
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
            fp16=(device=="cuda"),
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
        print("Saved tuned model to:", args.out_dir)


if __name__ == "__main__":
    main()
