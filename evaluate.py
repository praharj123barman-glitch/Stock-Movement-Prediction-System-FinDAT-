import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import config
from utils.data_loader import create_dataloaders
from models.fingat import FinGAT

def calculate_metrics(pred_returns, actual_returns, instance_idx, ks=[5, 10, 20]):
    """Calculate metrics for a single instance/day"""
    metrics_list = []
    
    # Get rankings for this day
    true_ranks = np.argsort(-actual_returns)
    pred_ranks = np.argsort(-pred_returns)
    
    for k in ks:
        # Create metrics for this K value
        metrics = {
            'K': k,
            'Instance_Ir': instance_idx
        }
        
        # Get top-K stocks for precision calculation
        true_top_k = set(true_ranks[:k].tolist())
        pred_top_k = set(pred_ranks[:k].tolist())
        
        # Calculate precision
        precision = len(true_top_k & pred_top_k) / k
        metrics['Precision'] = precision
        
        # Calculate MRR
        reciprocal_ranks = []
        for stock in true_top_k:
            rank = np.where(pred_ranks == stock)[0][0] + 1
            if rank <= k:
                reciprocal_ranks.append(1.0 / rank)
        metrics['MRR'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
        
        # Calculate IRR (Investment Return Ratio)
        true_top_k_indices = true_ranks[:k]  # Indices of actual top-K stocks
        pred_top_k_indices = pred_ranks[:k]  # Indices of predicted top-K stocks
        true_return = sum(actual_returns[idx] for idx in true_top_k_indices)
        pred_return = sum(actual_returns[idx] for idx in pred_top_k_indices)
        irr = (true_return - pred_return)  # Using absolute difference
        metrics['IRR'] = irr
        
        metrics_list.append(metrics)
    
    return metrics_list

def generate_test_predictions(model, test_loader, device):
    """Generate predictions with per-instance metrics"""
    model.eval()
    all_metrics = []
    
    # Get stock names and test dates
    sector_df = pd.read_csv(os.path.join(config.DATA_DIR, "sector_mapping.csv"))
    stock_names = sector_df['stock'].tolist()
    sample_stock = sector_df['stock'].iloc[0] + ".csv"
    df = pd.read_csv(os.path.join(config.CLEANED_DIR, sample_stock))
    test_dates = pd.to_datetime(df['Date']).iloc[config.TRAIN_DAYS+config.VAL_DAYS+config.WINDOW_SIZE:].tolist()
    
    instance_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate((test_loader)):
            features, adj_matrix, sectors, return_labels, _ = batch
            features = features.to(device)
            adj_matrix = adj_matrix.to(device)
            sectors = sectors.to(device)
            
            # Get predictions (batch_size, num_stocks)
            return_preds, _ = model(features, adj_matrix, sectors)
            preds = return_preds.cpu().numpy()
            actuals = return_labels.numpy()
            
            # Process each sample in batch
            for i in range(preds.shape[0]):
                day_idx = batch_idx * config.BATCH_SIZE + i
                if day_idx >= len(test_dates):
                    continue
                
                date_str = test_dates[day_idx].strftime("%Y-%m-%d")
                day_preds = preds[i].squeeze()
                day_actuals = actuals[i].squeeze()
                
                # Calculate metrics for this day/instance
                day_metrics = calculate_metrics(day_preds, day_actuals, instance_count)
                instance_count += 1
                
                # Add to overall metrics
                all_metrics.extend(day_metrics)
                
                # Create DataFrame for predictions
                df = pd.DataFrame({
                    "Date": [date_str] * len(stock_names),
                    "Stock": stock_names,
                    "Predicted_Return": day_preds.tolist() if hasattr(day_preds, 'tolist') else day_preds,
                    "Actual_Return": day_actuals.tolist() if hasattr(day_actuals, 'tolist') else day_actuals,
                    "Rank": np.argsort(-day_preds) + 1
                }).sort_values("Rank")
                
               
    # Save metrics in the desired format
    metrics_df = pd.DataFrame(all_metrics)
    
    # Reorder columns to match desired format
    metrics_df = metrics_df[['K', 'Instance_Ir', 'MRR', 'Precision', 'IRR']]
    
    # Print summary
    print("\nTest Metrics Summary:")
    for k in [5, 10, 20]:
        k_metrics = metrics_df[metrics_df['K'] == k]
        print(f"K={k}:")
        print(f"   Precision: {k_metrics['Precision'].mean():.4f}")
        print(f"   MRR: {k_metrics['MRR'].mean():.4f}")
        print(f"   IRR: {k_metrics['IRR'].mean():.4f}")

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(
        os.path.join(config.CHECKPOINTS_DIR, "best_model_dim8.pt"),
        map_location=device,
        weights_only=False
    )
    
    # Initialize model
    model = FinGAT(
        input_dim=16,
        hidden_dim=checkpoint['hidden_dim'],
        embed_dim=config.EMBEDDING_DIM,
        num_stocks=445,
        num_sectors=19,
        num_weeks=config.NUM_WEEKS,
        transformer_layers=config.TRANSFORMER_LAYERS,
        transformer_heads=config.TRANSFORMER_HEADS,
        dropout=0.1,
        delta=config.DELTA
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get test loader
    _, _, test_loader = create_dataloaders(batch_size=config.BATCH_SIZE)
    
    # Generate predictions
    generate_test_predictions(model, test_loader, device)

if __name__ == "__main__":
    main()
