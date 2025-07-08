# train.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from utils.data_loader import create_dataloaders
from utils.evaluation import evaluate_model
from models.fingat import FinGAT
from memory_efficient import gradient_accumulation_training, memory_cleanup

def main():
    """Train the FinGAT model with Dynamic Transformer"""
    print("Starting model training...")
    
    # Set device (use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=config.BATCH_SIZE)
    
    # Get a sample batch to determine dimensions
    sample_batch = next(iter(train_loader))
    stock_features, adj_matrix, sector_indices = sample_batch[:3]
    
    # Get dimensions from sample batch
    _, num_stocks, seq_len, input_dim = stock_features.size()
    num_sectors = len(torch.unique(sector_indices))
    
    print(f"Model parameters:")
    print(f"  - Number of stocks: {num_stocks}")
    print(f"  - Number of sectors: {num_sectors}")
    print(f"  - Input dimension: {input_dim}")
    print(f"  - GRU  dimension: {config.GRU_HIDDEN_DIM}")
    print(f"  - Embedding dimension: {config.EMBEDDING_DIM}")
    
    # Loop through different hidden dimensions to find best model
    best_val_mrr = 0.0
    best_hidden_dim = None
    
    for hidden_dim in config.HIDDEN_DIMS:
        print(f"\nTraining model with hidden_dim={hidden_dim}...")
        
        # Initialize model
        model = FinGAT(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embed_dim=config.EMBEDDING_DIM,
            num_stocks=num_stocks,
            num_sectors=num_sectors,
            num_weeks=config.NUM_WEEKS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_heads=config.TRANSFORMER_HEADS,
            dropout=0.1,
            delta=config.DELTA
        ).to(device)
        
        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.LAMBDA)
        
        # Create checkpoint directory
        os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
        print("NOw we start from epoch")
        # Train model
        for epoch in range(config.EPOCHS):
            print(f"\nStarting Epoch {epoch + 1}/{config.EPOCHS}")
            current_lr = optimizer.param_groups[0]['lr']
            print(f"🔧 Learning rate: {current_lr:.6f}")
            # Train with gradient accumulation for memory efficiency
            train_loss = gradient_accumulation_training(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                accumulation_steps=4,  # Adjust based on memory constraints
                device=device
            )
            
            print(f"Epoch {epoch+1}/{config.EPOCHS}, Loss: {train_loss:.4f}")
            
            # Evaluate on validation set
            val_metrics = evaluate_model(model, val_loader, device)
            
            print("Validation metrics:")
            for metric, value in val_metrics.items():
                print(f"  - {metric}: {value:.4f}")
            
            # Save checkpoint if validation MRR improves
            if val_metrics['mrr_5'] > best_val_mrr:
                best_val_mrr = val_metrics['mrr_5']
                best_hidden_dim = hidden_dim
                
                # Save checkpoint
                checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, f"best_model_dim{hidden_dim}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'hidden_dim': hidden_dim
                }, checkpoint_path)
                
                print(f"Saved new best model to {checkpoint_path}")
            
            # Clean up memory
            memory_cleanup()
    
    print(f"\nBest model has hidden_dim={best_hidden_dim} with validation MRR@5={best_val_mrr:.4f}")

if __name__ == "__main__":
    main()
