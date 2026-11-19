"""Retrain FinDAT with the corrected, differentiable ranking loss.

The original training loop called FinGAT.calculate_loss, whose ranking term only
adds label values (|true_diff|) and so has zero gradient w.r.t. the predictions.
This script trains with FinGAT.ranking_movement_loss instead, runs the hidden-size
ablation (8/16/32/64), selects each model on validation MRR@5, and reports
validation + test ranking metrics next to a random-ranking baseline.

New checkpoints go to checkpoints/fixed_loss/ (the original ones are kept).
Results are written to results/fixed_loss_metrics.json.

Usage:  python train_fixed_loss.py [--epochs 10] [--dims 8 16 32 64]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.optim as optim

import config
from models.fingat import FinGAT
from utils.data_loader import create_dataloaders
from utils.evaluation import evaluate_model, investment_return_ratio, mean_reciprocal_rank, precision_at_k

OUT_DIR = os.path.join(config.CHECKPOINTS_DIR, "fixed_loss")
RESULTS_PATH = os.path.join(config.RESULTS_DIR, "fixed_loss_metrics.json")


def train_one_epoch(model, loader, optimizer, device, accumulation_steps=4):
    model.train()
    total, steps = 0.0, 0
    optimizer.zero_grad()
    for i, (x, adj, sec, ret, mov) in enumerate(loader):
        x, adj, sec, ret, mov = (t.to(device) for t in (x, adj, sec, ret, mov))
        ret_pred, mov_pred = model(x, adj, sec)
        loss = model.ranking_movement_loss(ret_pred, mov_pred, ret, mov) / accumulation_steps
        loss.backward()
        total += loss.item() * accumulation_steps
        steps += 1
        if (i + 1) % accumulation_steps == 0 or i == len(loader) - 1:
            optimizer.step()
            optimizer.zero_grad()
    return total / max(steps, 1)


def random_baseline(loader, seed=0):
    """Metrics for a random ranking of the same days (the bar to beat)."""
    rng = np.random.default_rng(seed)
    labels = np.concatenate([b[3].numpy() for b in loader], axis=0)
    preds = rng.standard_normal(labels.shape)
    out = {}
    for k in (5, 10, 20):
        out[f"precision_{k}"] = float(precision_at_k(labels, preds, k))
        out[f"mrr_{k}"] = float(mean_reciprocal_rank(labels, preds, k))
        out[f"irr_{k}"] = float(investment_return_ratio(labels, preds, k))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--dims", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    train_loader, val_loader, test_loader = create_dataloaders(batch_size=config.BATCH_SIZE)
    x, _, sec = next(iter(train_loader))[:3]
    _, num_stocks, _, input_dim = x.size()
    num_sectors = len(torch.unique(sec))
    print(f"stocks={num_stocks} sectors={num_sectors} features={input_dim} device={device}", flush=True)

    results = {"loss": "ranking_movement_loss (differentiable pairwise hinge + BCE)",
               "epochs": args.epochs, "seed": args.seed,
               "random_baseline_test": random_baseline(test_loader), "models": {}}

    for dim in args.dims:
        torch.manual_seed(args.seed)
        model = FinGAT(input_dim=input_dim, hidden_dim=dim, embed_dim=config.EMBEDDING_DIM,
                       num_stocks=num_stocks, num_sectors=num_sectors, num_weeks=config.NUM_WEEKS,
                       transformer_layers=config.TRANSFORMER_LAYERS, transformer_heads=config.TRANSFORMER_HEADS,
                       dropout=0.1, delta=config.DELTA).to(device)
        opt = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.LAMBDA)
        best, best_epoch, path = -1.0, -1, os.path.join(OUT_DIR, f"best_model_dim{dim}.pt")
        for epoch in range(args.epochs):
            t0 = time.time()
            loss = train_one_epoch(model, train_loader, opt, device)
            val = evaluate_model(model, val_loader, device)
            print(f"dim={dim} epoch={epoch + 1}/{args.epochs} loss={loss:.4f} "
                  f"val_mrr5={val['mrr_5']:.4f} val_p5={val['precision_5']:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if val["mrr_5"] > best:
                best, best_epoch = val["mrr_5"], epoch + 1
                torch.save({"epoch": epoch, "hidden_dim": dim, "model_state_dict": model.state_dict(),
                            "val_metrics": {k: float(v) for k, v in val.items()}}, path)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test = evaluate_model(model, test_loader, device)
        results["models"][str(dim)] = {"best_epoch": best_epoch,
                                       "val": ckpt["val_metrics"],
                                       "test": {k: float(v) for k, v in test.items()}}
        print(f"dim={dim} TEST p5={test['precision_5']:.4f} mrr5={test['mrr_5']:.4f} "
              f"p20={test['precision_20']:.4f} move_acc={test['movement_accuracy']:.4f}", flush=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print("random baseline (test):", {k: round(v, 4) for k, v in results["random_baseline_test"].items()})
    print("saved", RESULTS_PATH)


if __name__ == "__main__":
    main()
