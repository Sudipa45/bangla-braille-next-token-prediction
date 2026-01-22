#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re, json, csv, random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

random.seed(42)

SEQ_LEN = 120
NUM_SAMPLES = 50000

# -------------------------------
# Bengali → Braille mapping
# (Simplified Bharati Braille)
# -------------------------------

BN2BR = {
    " ": "␣",

    "অ":"⠁","আ":"⠜","ই":"⠊","ঈ":"⠔","উ":"⠥","ঊ":"⠳","এ":"⠑","ঐ":"⠌","ও":"⠕","ঔ":"⠪",

    "ক":"⠅","খ":"⠭","গ":"⠛","ঘ":"⠣","ঙ":"⠬",
    "চ":"⠉","ছ":"⠡","জ":"⠚","ঝ":"⠵","ঞ":"⠻",
    "ট":"⠾","ঠ":"⠺","ড":"⠫","ঢ":"⠿","ণ":"⠼",
    "ত":"⠞","থ":"⠹","দ":"⠙","ধ":"⠮","ন":"⠝",
    "প":"⠏","ফ":"⠖","ব":"⠃","ভ":"⠧","ম":"⠍",
    "য":"⠽","র":"⠗","ল":"⠇","শ":"⠩","ষ":"⠯","স":"⠎","হ":"⠓",

    "া":"⠜","ি":"⠊","ী":"⠔","ু":"⠥","ূ":"⠳","ে":"⠑","ৈ":"⠌","ো":"⠕","ৌ":"⠪",
    "্":"⠈","ং":"⠰","ঃ":"⠠","ঁ":"⠄",

    "।":"⠲",",":"⠂","?":"⠦","!":"⠖","-":"⠤"
}

UNK = "⣿"

# -------------------------------
# Text cleaning
# -------------------------------

def clean_bn(text):
    text = re.sub(r"[^\u0980-\u09FF\s।?!,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------------
# Bangla → Braille
# -------------------------------

def bn_to_braille(text):
    return [BN2BR.get(ch, UNK) for ch in text]

# -------------------------------
# Sliding window generation
# -------------------------------

def make_samples(tokens):
    samples = []
    for i in range(len(tokens) - SEQ_LEN - 1):
        window = tokens[i:i+SEQ_LEN]
        target = tokens[i+SEQ_LEN]
        samples.append((window, target))
        if len(samples) >= NUM_SAMPLES:
            break
    return samples

# -------------------------------
# Main pipeline
# -------------------------------

def main():
    print("Loading Bengali Wikipedia from HuggingFace...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.bn", split="train")

    text = ""
    for row in tqdm(ds, total=len(ds)):
        text += " " + row["text"]

    print("Cleaning Bangla text...")
    text = clean_bn(text)

    print("Converting Bangla → Braille...")
    tokens = bn_to_braille(text)

    print("Generating sliding-window samples...")
    samples = make_samples(tokens)

    print("Total samples:", len(samples))

    # Build vocab
    vocab = {}
    for w,y in samples:
        for t in w + [y]:
            if t not in vocab:
                vocab[t] = len(vocab)

    def encode(seq):
        return [vocab[t] for t in seq]

    rows = []
    for w,y in samples:
        rows.append({
            "input_braille": " ".join(w),
            "target_braille": y,
            "input_ids": " ".join(map(str, encode(w))),
            "target_id": vocab[y]
        })

    random.shuffle(rows)

    n = len(rows)
    train = rows[:int(0.9*n)]
    val   = rows[int(0.9*n):int(0.95*n)]
    test  = rows[int(0.95*n):]

    out = Path("bangla_braille_lm_20k")
    out.mkdir(exist_ok=True)

    def save_csv(name, data):
        with open(out/name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(data)

    save_csv("train.csv", train)
    save_csv("val.csv", val)
    save_csv("test.csv", test)
    save_csv("all.csv", rows)

    with open(out/"vocab.json","w",encoding="utf-8") as f:
        json.dump(vocab,f,ensure_ascii=False,indent=2)

    print("Dataset saved in:", out.resolve())

if __name__ == "__main__":
    main()
