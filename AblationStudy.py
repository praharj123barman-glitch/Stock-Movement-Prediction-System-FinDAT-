# ablation_study.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import config
from utils.data_loader import create_dataloaders
from utils.evaluation import evaluate_model
from models.fingat import FinGAT
import gc


class AblationFinGAT(FinGAT):
    """
    FinGAT model with options to disable components for ablation study
    """
    def __init__(self, input_dim, hidden_dim, embed_dim, num_stocks, num_sectors,
                num_weeks=4, transformer_layers=2, transformer_heads=4, dropout=0.1, delta=0.5,
                use_transformer=True, use_sector_model=True, use_long_term=True):
        
        super(AblationFinGAT, self).__init__(
            input_dim, hidden_dim, embed_dim, num_stocks, num_sectors,
            num_weeks, transformer_layers, transformer_heads, dropout, delta
        )
        
        self.use_transformer = use_transformer
        self.use_sector_model = use_sector_model
        self.use_long_term = use_long_term
    
    def forward(self, stock_features, adj_matrix, sector_indices, historical_embeddings=None):
        """
        Modified forward pass to enable/disable components for ablation study
        """
        batch_size, num_stocks, seq_len, _ = stock_features.size()
        
        # Short-term sequential learning for each stock
        stock_embeddings = torch.zeros(batch_size, num_stocks, self.embed_dim).to(stock_features.device)
        for i in range(num_stocks):
            stock_hidden = self.short_term_gru(stock_features[:, i])
            stock_embeddings[:, i] = self.stock_proj(stock_hidden)
        
        # Intra-sector relation modeling with dynamic transformer 
        if self.use_transformer:
            intra_sector_embeddings = self.intra_sector_transformer(stock_embeddings, adj_matrix)
        else:
            intra_sector_embeddings = stock_embeddings
        
        # Long-term sequential learning 
        if self.use_long_term and historical_embeddings is not None:
            long_term_embeddings = torch.zeros(batch_size, num_stocks, self.embed_dim).to(stock_features.device)
            for i in range(num_stocks):
                long_term_hidden = self.long_term_gru(historical_embeddings[:, i])
                long_term_embeddings[:, i] = self.long_term_proj(long_term_hidden)
        else:
            # If disabled or no historical data, initialize with zeros
            long_term_embeddings = torch.zeros(batch_size, num_stocks, self.embed_dim).to(stock_features.device)
        
        # Sector-level modeling 
        if self.use_sector_model:
            sector_embeddings = self.sector_model(intra_sector_embeddings, sector_indices)
        else:
            sector_embeddings = torch.zeros(batch_size, num_stocks, self.embed_dim).to(stock_features.device)
        
        if self.use_transformer and self.use_sector_model and self.use_long_term:
            fused_embeddings = self.fusion(
                torch.cat([intra_sector_embeddings, long_term_embeddings, sector_embeddings], dim=2)
            )
        elif self.use_transformer and self.use_sector_model:
            fused_embeddings = self.fusion(
                torch.cat([intra_sector_embeddings, torch.zeros_like(long_term_embeddings), sector_embeddings], dim=2)
            )
        elif self.use_transformer and self.use_long_term:
            fused_embeddings = self.fusion(
                torch.cat([intra_sector_embeddings, long_term_embeddings, torch.zeros_like(sector_embeddings)], dim=2)
            )
        elif self.use_sector_model and self.use_long_term:

            fused_embeddings = self.fusion(
                torch.cat([stock_embeddings, long_term_embeddings, sector_embeddings], dim=2)
            )
        elif self.use_transformer:

            fused_embeddings = self.fusion(
                torch.cat([intra_sector_embeddings, 
                           torch.zeros_like(long_term_embeddings), 
                           torch.zeros_like(sector_embeddings)], dim=2)
            )
        elif self.use_sector_model:
            fused_embeddings = self.fusion(
                torch.cat([stock_embeddings, 
                           torch.zeros_like(long_term_embeddings), 
                           sector_embeddings], dim=2)
            )
        elif self.use_long_term:
            fused_embeddings = self.fusion(
                torch.cat([stock_embeddings, 
                           long_term_embeddings, 
                           torch.zeros_like(sector_embeddings)], dim=2)
            )
        else:
            zeros1 = torch.zeros_like(stock_embeddings)
            zeros2 = torch.zeros_like(stock_embeddings)
            fused_embeddings = self.fusion(
                torch.cat([stock_embeddings, zeros1, zeros2], dim=2)
            )
        
        # Task-specific predictions
        return_predictions = self.return_predictor(fused_embeddings).squeeze(-1)
        movement_predictions = self.movement_predictor(fused_embeddings).squeeze(-1)
        
        return return_predictions, movement_predictions


def train_and_evaluate(model, train_loader, val_loader, test_loader, device, epochs=1):
    """Train and evaluate a model"""
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.LAMBDA)
    
    best_val_mrr = 0.0
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            stock_features, adj_matrix, sector_indices, return_labels, movement_labels = batch
            
            # Move to device
            stock_features = stock_features.to(device)
            adj_matrix = adj_matrix.to(device)
            sector_indices = sector_indices.to(device)
            return_labels = return_labels.to(device)
            movement_labels = movement_labels.to(device)
            
            # Forward pass
            return_preds, movement_preds = model(stock_features, adj_matrix, sector_indices)
            
            # Calculate loss
            loss = model.calculate_loss(return_preds, movement_preds, return_labels, movement_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluate on validation set
        val_metrics = evaluate_model(model, val_loader, device)
        test_metrics=evaluate_model(model,test_loader,device)

        print("Validation metrics:")
        for metric, value in val_metrics.items():
            print(f" - {metric}: {value:.4f}")
        
        print("Test metrics:")
        for metric, value in test_metrics.items():
            print(f" - {metric}: {value:.4f}")

        # Save best model based on MRR@5
        if val_metrics['mrr_5'] > best_val_mrr:
            best_val_mrr = val_metrics['mrr_5']
            best_model_state = model.state_dict().copy()
    
    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    

def gradient_accumulation_training(model, train_loader, optimizer, accumulation_steps=4, device='cuda'):
    """
    Train with gradient accumulation for memory efficiency
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Set gradients to zero before starting
    optimizer.zero_grad()
    
    for i, batch in enumerate(train_loader):
        stock_features, adj_matrix, sector_indices, return_labels, movement_labels = batch
        
        # Move to device
        stock_features = stock_features.to(device)
        adj_matrix = adj_matrix.to(device)
        sector_indices = sector_indices.to(device)
        return_labels = return_labels.to(device)
        movement_labels = movement_labels.to(device)
        
        # Forward pass
        return_preds, movement_preds = model(stock_features, adj_matrix, sector_indices)
        
        # Calculate loss and normalize by accumulation steps
        loss = model.calculate_loss(return_preds, movement_preds, return_labels, movement_labels)
        loss = loss / accumulation_steps
        
        # Backward pass (accumulate gradients)
        loss.backward()
        
        # Update parameters only after accumulation_steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps 
        num_batches += 1
        
        # Free up memory
        del stock_features, adj_matrix, sector_indices, return_labels, movement_labels
        del return_preds, movement_preds, loss
        torch.cuda.empty_cache() if device == 'cuda' else None
    
    avg_loss = total_loss / num_batches
    return avg_loss


def run_ablation_study():
    """Run complete ablation study"""
    print("Starting ablation study...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=config.BATCH_SIZE)
    
    # Get dimensions from a sample batch
    sample_batch = next(iter(train_loader))
    stock_features, adj_matrix, sector_indices = sample_batch[:3]
    _, num_stocks, seq_len, input_dim = stock_features.size()
    num_sectors = len(torch.unique(sector_indices))
    
    print(f"Model parameters:")
    print(f" - Number of stocks: {num_stocks}")
    print(f" - Number of sectors: {num_sectors}")
    print(f" - Input dimension: {input_dim}")
    
    # Define model variants for ablation study
    model_variants = {
        "Short Term GRU + Dynamic Transformer": AblationFinGAT(
            input_dim=input_dim,
            hidden_dim=config.GRU_HIDDEN_DIM,
            embed_dim=config.EMBEDDING_DIM,
            num_stocks=num_stocks,
            num_sectors=num_sectors,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_heads=config.TRANSFORMER_HEADS,
            dropout=0.1,
            delta=config.DELTA,
            use_transformer=True,
            use_sector_model=False,
            use_long_term=False
        ),
        "Short term GRU + SectorModel": AblationFinGAT(
            input_dim=input_dim,
            hidden_dim=config.GRU_HIDDEN_DIM,
            embed_dim=config.EMBEDDING_DIM,
            num_stocks=num_stocks,
            num_sectors=num_sectors,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_heads=config.TRANSFORMER_HEADS,
            dropout=0.1,
            delta=config.DELTA,
            use_transformer=False,
            use_sector_model=True,
            use_long_term=False
        ),
        "Short Term GRU + LongTerm": AblationFinGAT(
            input_dim=input_dim,
            hidden_dim=config.GRU_HIDDEN_DIM,
            embed_dim=config.EMBEDDING_DIM,
            num_stocks=num_stocks,
            num_sectors=num_sectors,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_heads=config.TRANSFORMER_HEADS,
            dropout=0.1,
            delta=config.DELTA,
            use_transformer=False,
            use_sector_model=False,
            use_long_term=True
        ),
        
    }
    
    # Store results
    results = {}
    
    # Train and evaluate each model variant
    for model_name, model in model_variants.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        model = model.to(device)
        metrics = train_and_evaluate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=1
        )
        
        results[model_name] = metrics
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).T
    
    # Save results
    os.makedirs("Ablationresults", exist_ok=True)
    results_df.to_csv("Ablationresults/ablation_study_results.csv")
    
    # Plot key metrics
    plot_key_metrics(results_df)
    
    return results_df


def plot_key_metrics(results_df):
    """Plot key metrics from ablation study"""
    key_metrics = ['precision_5', 'precision_10', 'mrr_5', 'mrr_10', 'movement_accuracy']
    
    # Bar chart for each key metric
    plt.figure(figsize=(15, 12))
    for i, metric in enumerate(key_metrics):
        plt.subplot(len(key_metrics), 1, i+1)
        results_df[metric].plot(kind='bar', color='skyblue')
        plt.title(metric)
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig("Ablationresults/ablation_metrics.png")
    
    # Heatmap of all metrics
    plt.figure(figsize=(15, 10))
    sns.heatmap(results_df, annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title('Ablation Study Results')
    plt.tight_layout()
    plt.savefig("Ablationresults/ablation_heatmap.png")
    
    # Create summary table for key metrics
    key_df = results_df[['precision_5', 'mrr_5', 'movement_accuracy']]
    
    

if __name__ == "__main__":

    
    # Run the ablation study
    ablation_results = run_ablation_study()
    
    # Print summary of results
    print("\nSummary of Ablation Study Results:")
    print(ablation_results[['precision_5', 'mrr_5', 'irr_5', 'movement_accuracy']])
