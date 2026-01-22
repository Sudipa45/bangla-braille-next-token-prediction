# Bangla Braille Next-Token Prediction

This repository provides a complete framework for **Bangla Braille next-token prediction** using
neural sequence models, with a focus on **assistive Braille typing systems**.

The project includes:
- A Bengali Wikipedia–derived Braille dataset
- A BiLSTM baseline
- An enhanced **BiLSTM + Attention** model
- Comprehensive evaluation metrics tailored to assistive typing

---

## 🔍 Task Description

Given a fixed-length context of Bangla Braille characters, the model predicts the **next Braille token**.

This formulation supports:
- Assistive Braille input methods
- Keystroke saving through top-K suggestions
- Language modeling for low-resource accessibility tools

---

## 📊 Dataset

The dataset is derived from **Bengali Wikipedia**, converted into Bangla Braille and formatted using
a sliding-window approach. Dataset Link: https://drive.google.com/drive/folders/1Q8oo2AvH-j0HUK1gG1Ew80W3-wMQSi-X?usp=sharing

Each sample consists of:
- `input_ids`: sequence of Braille token IDs (length = 30)
- `target_id`: next Braille token ID

Due to size constraints, raw data is not included.
See `data/README.md` for generation instructions.

---

## 🧠 Models

### 1️⃣ BiLSTM Baseline
- Character-level embeddings
- Bidirectional LSTM
- Final hidden-state classification

### 2️⃣ BiLSTM + Attention (Main Model)
- Temporal attention over BiLSTM outputs
- Context-aware sequence aggregation
- Improved ranking and keystroke efficiency

---

## 📐 Evaluation Metrics

We report metrics suitable for **assistive typing systems**:

- **Top-K Accuracy (K = 1, 3, 5)**
- **Keystroke Saving Rate (KSR@K)**
- **Mean Reciprocal Rank (MRR)**
- **Coverage Error** (mean rank of correct token)
- **Cross-Entropy Loss**
- **Perplexity**

---

## 🚀 Training

### Train BiLSTM + Attention with logging

```
python scripts/braille_bilstm_attn_metrics_logsave.py \
  --data_dir /path/to/bangla_braille_lm_20k \
  --epochs 20 \
  --batch_size 128 \
  --lr 2e-3 \
  --label_smoothing 0.05 \
  --out_dir runs/bilstm_attn_ls
```
