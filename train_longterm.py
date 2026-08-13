"""Train FinGAT with the long-term Stage 3 branch ACTIVE.

Why this file exists
--------------------
In the shipped model the long-term Attentive GRU is wired but never fed: every
caller invoked `model(features, adj, sector)` with no history, so the branch
output zeros and one third of the fusion was dead. This entry point:

  1. asks the dataloader for the last config.NUM_WEEKS windows per stock
     (create_dataloaders(..., include_history=True)),
  2. passes them into forward() as `historical_windows`, which activates the
     long-term Attentive GRU (see models/fingat.py), and
  3. trains with the corrected, vectorized ranking loss
     (model.ranking_movement_loss) instead of the zero-gradient original.

Run:  python train_longterm.py
"""
import os
import copy
import numpy as np
import torch
import torch.optim as optim

import config
from utils.data_loader import create_dataloaders
from utils.evaluation import (precision_at_k, mean_reciprocal_rank,
                              investment_return_ratio)
from models.fingat import FinGAT

EMBED_DIM = config.EMBEDDING_DIM   # 8
HIDDEN_DIM = 64                    # GRU / long-term hidden size
EPOCHS = config.EPOCHS
MARGIN = 0.05

torch.manual_seed(42)


@torch.no_grad()
def evaluate(model, loader, device):
    """Collect daily rankings and compute Precision@5 / MRR@5 / IRR@5."""
    model.eval()
    preds, labels = [], []
    for batch in loader:
        features, adj, sector, ret_labels, _mov_labels, history = batch
        rp, _ = model(features.to(device), adj.to(device),
                      sector.to(device), history.to(device))
        preds.append(rp.cpu().numpy())
        labels.append(ret_labels.numpy())
    preds = np.concatenate(preds, axis=0)     # (num_days, num_stocks)
    labels = np.concatenate(labels, axis=0)
    return {
        "p5": precision_at_k(labels, preds, 5),
        "mrr5": mean_reciprocal_rank(labels, preds, 5),
        "irr5": investment_return_ratio(labels, preds, 5),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # include_history=True -> each batch carries the last NUM_WEEKS windows.
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=config.BATCH_SIZE, include_history=True)

    # Infer dimensions from one batch.
    features, adj, sector, ret_labels, mov_labels, history = next(iter(train_loader))
    _, num_stocks, seq_len, input_dim = features.size()
    num_sectors = len(torch.unique(sector))
    print(f"stocks={num_stocks} sectors={num_sectors} input_dim={input_dim} "
          f"weeks={history.size(2)}")

    model = FinGAT(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        num_stocks=num_stocks,
        num_sectors=num_sectors,
        num_weeks=config.NUM_WEEKS,
        transformer_layers=config.TRANSFORMER_LAYERS,
        transformer_heads=config.TRANSFORMER_HEADS,
        dropout=0.1,
        delta=config.DELTA,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                           weight_decay=config.LAMBDA)

    best_p5 = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        model.train()
        running, nb = 0.0, 0
        for batch in train_loader:
            features, adj, sector, ret_labels, mov_labels, history = batch
            features, adj, sector = features.to(device), adj.to(device), sector.to(device)
            ret_labels, mov_labels, history = (ret_labels.to(device),
                                               mov_labels.to(device), history.to(device))

            rp, mp = model(features, adj, sector, history)   # Stage 3 ACTIVE
            loss = model.ranking_movement_loss(rp, mp, ret_labels, mov_labels, margin=MARGIN)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            nb += 1

        vm = evaluate(model, val_loader, device)
        print(f"epoch {epoch+1}/{EPOCHS}  loss {running/max(nb,1):.4f}  "
              f"valP@5 {vm['p5']:.4f}  valMRR@5 {vm['mrr5']:.4f}  valIRR@5 {vm['irr5']:.4f}")
        if vm["p5"] > best_p5:
            best_p5 = vm["p5"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    tm = evaluate(model, test_loader, device)
    print("\n=== LONG-TERM (Stage 3) ACTIVE, TEST split ===")
    print(f"Precision@5 {tm['p5']:.4f}  MRR@5 {tm['mrr5']:.4f}  IRR@5 {tm['irr5']:.4f}")

    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    out = os.path.join(config.CHECKPOINTS_DIR, "longterm_active.pt")
    torch.save({"hidden_dim": HIDDEN_DIM, "embed_dim": EMBED_DIM,
                "model_state_dict": best_state}, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
