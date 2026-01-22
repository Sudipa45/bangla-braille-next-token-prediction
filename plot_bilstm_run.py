#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
import pandas as pd
import matplotlib.pyplot as plt

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to run folder containing epoch_log.csv etc.")
    args = ap.parse_args()

    run_dir = args.run_dir
    csv_path = os.path.join(run_dir, "epoch_log.csv")
    final_path = os.path.join(run_dir, "final_metrics.json")
    cfg_path = os.path.join(run_dir, "run_config.json")

    df = pd.read_csv(csv_path)

    # Ensure numeric
    for c in df.columns:
        if c != "epoch":
            df[c] = df[c].apply(safe_float)

    # Best epoch by val_top1
    best_idx = df["val_top1"].idxmax()
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_val = float(df.loc[best_idx, "val_top1"])

    # Print a tiny summary
    print(f"[Best] epoch={best_epoch} val_top1={best_val:.4f}")

    # Load extra info (optional)
    if os.path.exists(final_path):
        with open(final_path, "r", encoding="utf-8") as f:
            final = json.load(f)
        print("[Final metrics]")
        print(json.dumps(final, indent=2, ensure_ascii=False))

    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        hp = cfg.get("hyperparameters", {})
        print("\n[Hyperparameters]")
        for k in sorted(hp.keys()):
            print(f"{k}: {hp[k]}")

    def vline_best():
        plt.axvline(best_epoch, linestyle="--")

    # 1) Train loss
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"])
    vline_best()
    plt.xlabel("Epoch"); plt.ylabel("Train Loss")
    plt.title("Training Loss")
    out = os.path.join(run_dir, "plot_train_loss.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

    # 2) Val Top-K
    plt.figure()
    plt.plot(df["epoch"], df["val_top1"], label="Top-1")
    plt.plot(df["epoch"], df["val_top3"], label="Top-3")
    plt.plot(df["epoch"], df["val_top5"], label="Top-5")
    vline_best()
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Validation Top-K Accuracy")
    plt.legend()
    out = os.path.join(run_dir, "plot_val_topk.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

    # 3) Val MRR
    plt.figure()
    plt.plot(df["epoch"], df["val_mrr"])
    vline_best()
    plt.xlabel("Epoch"); plt.ylabel("MRR")
    plt.title("Validation Mean Reciprocal Rank (MRR)")
    out = os.path.join(run_dir, "plot_val_mrr.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

    # 4) Val Coverage Error
    plt.figure()
    plt.plot(df["epoch"], df["val_coverage"])
    vline_best()
    plt.xlabel("Epoch"); plt.ylabel("Mean Rank (lower better)")
    plt.title("Validation Coverage Error")
    out = os.path.join(run_dir, "plot_val_coverage.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

    # 5) Val Perplexity
    plt.figure()
    plt.plot(df["epoch"], df["val_ppl"])
    vline_best()
    plt.xlabel("Epoch"); plt.ylabel("Perplexity (lower better)")
    plt.title("Validation Perplexity")
    out = os.path.join(run_dir, "plot_val_ppl.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

    # 6) Val mean CE
    plt.figure()
    plt.plot(df["epoch"], df["val_mean_ce"])
    vline_best()
    plt.xlabel("Epoch"); plt.ylabel("Mean Cross-Entropy")
    plt.title("Validation Mean Cross-Entropy")
    out = os.path.join(run_dir, "plot_val_mean_ce.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

    # Save a small “best row” summary CSV
    best_row = df.loc[[best_idx]]
    best_csv = os.path.join(run_dir, "best_epoch_row.csv")
    best_row.to_csv(best_csv, index=False)
    print("Saved:", best_csv)

if __name__ == "__main__":
    main()
