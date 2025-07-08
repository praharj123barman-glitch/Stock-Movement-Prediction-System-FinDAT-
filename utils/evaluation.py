# utils/evaluation.py
import numpy as np
import pandas as pd
import torch
def precision_at_k(true_returns, pred_returns, k=5):
    """
    Calculate Precision@K for top-K stocks per day
    true_returns: 2D array (num_days, num_stocks)
    pred_returns: 2D array (num_days, num_stocks)
    """
    precision_scores = []
    
    for day_true, day_pred in zip(true_returns, pred_returns):
        # Get indices of top K true and predicted stocks
        true_top_k = set(np.argsort(-day_true)[:k].tolist())
        pred_top_k = set(np.argsort(-day_pred)[:k].tolist())
        
        # Calculate precision for this day
        precision = len(pred_top_k & true_top_k) / k
        precision_scores.append(precision)
    
    return np.mean(precision_scores)

def mean_reciprocal_rank(true_returns, pred_returns, k=5):
    """Calculate MRR@K per day"""
    mrr_scores = []
    
    for day_true, day_pred in zip(true_returns, pred_returns):
        # Get true top K stocks
        true_top = np.argsort(-day_true)[:k]
        
        # Get predicted rankings
        pred_ranks = np.argsort(-day_pred)
        
        day_mrr = 0.0
        for stock_idx in true_top:
            # Find rank position (1-based index)
            rank = np.where(pred_ranks == stock_idx)[0][0] + 1
            day_mrr += 1.0 / rank
        
        mrr_scores.append(day_mrr / k)
    
    return np.mean(mrr_scores)

def investment_return_ratio(true_returns, pred_returns, k=5):
    """Calculate IRR@K per day"""
    irr_scores = []
    
    for day_true, day_pred in zip(true_returns, pred_returns):
        # Get true and predicted top K indices
        true_top = np.argsort(-day_true)[:k]
        pred_top = np.argsort(-day_pred)[:k]
        
        # Calculate returns
        true_return = np.sum(day_true[true_top])
        pred_return = np.sum(day_true[pred_top])
        
        irr_scores.append(true_return - pred_return)
    
    return np.mean(irr_scores)

def evaluate_model(model, data_loader, device):
    """Evaluate model with proper daily rankings"""
    model.eval()
    all_return_preds = []
    all_movement_preds = []
    all_return_labels = []
    all_movement_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            stock_features, adj_matrix, sector_indices, return_labels, movement_labels = batch
            
            # Move to device
            stock_features = stock_features.to(device)
            adj_matrix = adj_matrix.to(device)
            sector_indices = sector_indices.to(device)
            
            # Forward pass (batch_size, num_stocks)
            return_preds, movement_preds = model(stock_features, adj_matrix, sector_indices)
            
            # Store predictions and labels (keep batch dimension)
            all_return_preds.append(return_preds.cpu().numpy())
            all_return_labels.append(return_labels.cpu().numpy())
            all_movement_preds.append(movement_preds.cpu().numpy())
            all_movement_labels.append(movement_labels.cpu().numpy())
    
    # Concatenate while preserving batch (day) dimension
    return_preds = np.concatenate(all_return_preds, axis=0)  # (num_days, num_stocks)
    return_labels = np.concatenate(all_return_labels, axis=0)
    movement_preds = np.concatenate(all_movement_preds, axis=0)
    movement_labels = np.concatenate(all_movement_labels, axis=0)
    
    # Calculate metrics
    metrics = {
        'mrr_5': mean_reciprocal_rank(return_labels, return_preds, k=5),
        'mrr_10': mean_reciprocal_rank(return_labels, return_preds, k=10),
        'mrr_20': mean_reciprocal_rank(return_labels, return_preds, k=20),
        'precision_5': precision_at_k(return_labels, return_preds, k=5),
        'precision_10': precision_at_k(return_labels, return_preds, k=10),
        'precision_20': precision_at_k(return_labels, return_preds, k=20),
        'irr_5': investment_return_ratio(return_labels, return_preds, k=5),
        'irr_10': investment_return_ratio(return_labels, return_preds, k=10),
        'irr_20': investment_return_ratio(return_labels, return_preds, k=20),
        'movement_accuracy': np.mean((movement_preds > 0.5) == movement_labels)
    }
    
    return metrics
