import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import config
from models.fingat import FinGAT

def is_weekday(date):
    """Check if date is a weekday (Monday-Friday)"""
    return date.weekday() < 5  # 0=Monday, 4=Friday

def preprocess_ohlcv_data(data_dir, window_size=5):
    """Process stock data with robust handling of missing dates/values"""
    stock_data = []
    stock_names = []
    
    full_dates = pd.date_range(start="2025-01-11", end="2025-04-10", freq='B')
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for csv_file in tqdm(csv_files, desc="Processing stocks"):
        try:
            # Read CSV with error handling
            df = pd.read_csv(os.path.join(data_dir, csv_file))
            
            # Ensure required columns exist and convert dates
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            
            # Filter out weekends and ensure we have data up to March 22
            df = df[(df['Date'].dt.weekday < 5) & (df['Date'] <= pd.to_datetime("2025-04-10"))]
            
            # Reindex to full business date range and fill missing values
            df = df.set_index('Date').reindex(full_dates).reset_index()
            df['Sector'] = df['Sector'].ffill().bfill().fillna("Unknown")
            
            # Fill numerical columns
            num_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[num_cols] = df[num_cols].fillna(0)
            
            # Add features with missing data handling
            df = add_ohlcv_features(df)
            
            # Create windows with padding
            windows = create_windows(df, window_size)
            stock_data.append(windows)
            stock_names.append(os.path.splitext(csv_file)[0])
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    return stock_data, stock_names

def add_ohlcv_features(df):
    """Add technical indicators with NaN handling"""
    df = df.copy()
    
    # Calculate returns with zero-filling
    df['return_ratio'] = df['Close'].pct_change().fillna(0)
    
    # Percentage changes with forward filling
    for col in ['Open', 'High', 'Low']:
        df[f'pct_change_{col.lower()}'] = df[col].pct_change().fillna(0)
    
    # Moving averages with minimum periods=1
    for window in [5, 10, 15, 20, 25, 30]:
        df[f'ma_{window}'] = df['Close'].rolling(window, min_periods=1).mean().fillna(0)
    
    # Binary movement indicator
    df['up_or_down'] = (df['Close'].diff() > 0).astype(int).fillna(0)
    
    return df

def create_windows(df, window_size=5):
    """Create sliding windows with padding"""
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    windows = []
    
    # Create windows even if data is shorter than window_size
    for i in range(len(df)):
        start = max(0, i - window_size + 1)
        end = i + 1
        window = df.iloc[start:end][feature_cols]
        
        # Pad with zeros if window is smaller than window_size
        if len(window) < window_size:
            padding = np.zeros((window_size - len(window), len(feature_cols)))
            window = np.vstack([padding, window.values])
        else:
            window = window.values
            
        windows.append(window[-window_size:])  # Take last window_size elements
    
    return np.array(windows, dtype=np.float32)

def calculate_metrics(pred_returns, actual_returns, instance_idx, ks=[5, 10, 20]):
    """Calculate metrics for a single instance/day"""
    metrics_list = []
    
    # Ensure inputs are properly shaped arrays
    pred_returns = np.asarray(pred_returns).squeeze()
    actual_returns = np.asarray(actual_returns).squeeze()
    
    # Get rankings for this day
    true_ranks = np.argsort(-actual_returns)
    pred_ranks = np.argsort(-pred_returns)
    
    for k in ks:
        # Create metrics for this K value
        metrics = {
            'K': k,
            'Instance_Ir': instance_idx
        }
        
        # Get top K stocks
        true_top_k = set(true_ranks[:k])
        pred_top_k = set(pred_ranks[:k])

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
        true_top_k_indices = true_ranks[:k]
        pred_top_k_indices = pred_ranks[:k]
        true_return = sum(actual_returns[idx] for idx in true_top_k_indices)
        pred_return = sum(actual_returns[idx] for idx in pred_top_k_indices)
        metrics['IRR'] = (true_return - pred_return)
        
        metrics_list.append(metrics)
    
    return metrics_list

def get_next_day_data(date):
    """Get stock data for the next business day after the given date"""
    # Find the next business day (skip weekends)
    next_date = date + pd.Timedelta(days=1)
    while next_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_date += pd.Timedelta(days=1)
    
    # Don't look beyond March 22
    if next_date > pd.to_datetime("2025-04-10"):
        return pd.DataFrame({'stock': [], 'return_ratio': []}).set_index('stock')
    
    # Create a dictionary to store return ratios by stock name
    return_ratios = {}
    
    # Process each stock file
    for csv_file in os.listdir(config.CLEANED_DIR):
        if not csv_file.endswith('.csv'):
            continue
            
        # Get stock name from filename
        stock_name = os.path.splitext(csv_file)[0]
        
        # Read CSV file
        df = pd.read_csv(os.path.join(config.CLEANED_DIR, csv_file))
        df['Date'] = pd.to_datetime(df['Date'])
        
        df = df[(df['Date'].dt.weekday < 5) & (df['Date'] <= pd.to_datetime("2025-04-10"))]
        
        # Find data for next trading day
        next_day_row = df[df['Date'] == next_date]
        
        # If we have data for the next day, calculate return ratio
        if not next_day_row.empty:
            # Calculate return ratio if not already in the data
            if 'return_ratio' not in next_day_row.columns:
                # Get current and previous day's close price
                current_close = next_day_row['Close'].values[0]
                
                # Find previous day's row (business day)
                prev_day_row = df[df['Date'] < next_date].tail(1)
                if not prev_day_row.empty:
                    prev_close = prev_day_row['Close'].values[0]
                    # Calculate return ratio
                    return_ratio = (current_close - prev_close) / prev_close
                else:
                    return_ratio = 0.0  # No previous data
            else:
                return_ratio = next_day_row['return_ratio'].values[0]
                
            # Store the return ratio
            return_ratios[stock_name] = return_ratio
        else:
            # No data for next day, use zero
            return_ratios[stock_name] = 0.0
    
    # Convert dictionary to pandas DataFrame for easier handling
    return pd.DataFrame({
        'stock': list(return_ratios.keys()),
        'return_ratio': list(return_ratios.values())
    }).set_index('stock')


def generate_daily_predictions(model, data_loader, device, stock_names, dates):
    """Generate predictions and metrics for each business day"""
    model.eval()
    all_metrics = []
    all_rankings = []
    
    # Filter dates to only include weekdays up to March 22
    business_dates = [date for date in dates if is_weekday(date) and date <= pd.to_datetime("2025-04-10")]
    
    with torch.no_grad():
        # We need to track both the batch index and business day index
        business_day_idx = 0
        
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Generating predictions")):
            # Stop if we've processed all business days
            if business_day_idx >= len(business_dates):
                break
                
            current_date = business_dates[business_day_idx]
            
            features, adj_matrix, sectors = batch[:3]
            features = features.to(device)
            adj_matrix = adj_matrix.to(device)
            sectors = sectors.to(device)
            
            # Forward pass to get predictions
            return_preds, _ = model(features, adj_matrix, sectors)
            preds = return_preds.cpu().numpy().squeeze()
            
            if business_day_idx < len(business_dates) - 1:
                next_business_date = business_dates[business_day_idx + 1]
                
                # Get actual returns from the next business day's data
                next_day_data = get_next_day_data(current_date)
                actuals = next_day_data['return_ratio'].values
                
                if len(actuals) == len(preds):  # Only calculate if we have matching data
                    # Calculate metrics for this day
                    day_metrics = calculate_metrics(preds, actuals, business_day_idx)
                    all_metrics.extend(day_metrics)
                    
                    if len(preds.shape) > 1:
                        preds = preds.reshape(-1)
                    
                    # Save daily rankings
                    ranks = np.argsort(-preds) + 1
                    df = pd.DataFrame({
                        "Date": [current_date.strftime("%Y-%m-%d")] * len(stock_names),
                        "Stock": stock_names,
                        "Predicted_Return": preds.tolist(),
                        "Rank": ranks.tolist()
                    }).sort_values("Rank")
                    
                    all_rankings.append(df)
                    
                    # Save the daily ranking to CSV
                    filename = f"Predicted_{current_date.strftime('%Y-%m-%d')}.csv"
                    df.to_csv(os.path.join(config.RESULTS_DIR, filename), index=False)
            
            business_day_idx += 1
    
    # Save metrics to CSV
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(os.path.join(config.RESULTS_DIR, "instance_metrics.csv"), index=False)
    
    return all_metrics, all_rankings
class StockDataset(torch.utils.data.Dataset):
    def __init__(self, stock_data, stock_names, sector_indices):
        self.stock_data = stock_data
        self.stock_names = stock_names
        self.sector_indices = sector_indices
        self.num_stocks = len(stock_names)
        
        # Use maximum possible windows across all stocks
        self.window_indices = range(max(len(data) for data in stock_data))
    
    def __getitem__(self, idx):
        batch_features = []
        for i in range(self.num_stocks):
            # Handle stocks with insufficient data
            if idx < len(self.stock_data[i]):
                features = self.stock_data[i][idx]
            else:
                # Create zero-padded dummy data
                features = np.zeros_like(self.stock_data[i][0])
                
            batch_features.append(features)
        
        # Create adjacency matrix
        adj_matrix = np.zeros((self.num_stocks, self.num_stocks), dtype=np.float32)
        for i in range(self.num_stocks):
            for j in range(self.num_stocks):
                if i != j and self.sector_indices[i] == self.sector_indices[j]:
                    adj_matrix[i, j] = 1.0
        
        return (
            torch.tensor(np.array(batch_features)),
            torch.tensor(adj_matrix),
            torch.tensor(self.sector_indices)
        )
    
    def __len__(self):
        return len(self.window_indices)

def create_data_loader(stock_data, stock_names, sector_indices, batch_size=1):
    """Create dataloader for stock inference"""
    # Create dataset
    dataset = StockDataset(stock_data, stock_names, sector_indices)
    
    # Create and return DataLoader
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

def main():
    """Main function to generate instance metrics and rankings"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Define date range (January 11, 2025 to March 22, 2025)
    all_dates = pd.date_range(start="2025-01-11", end="2025-04-10")
    
    # 1. Preprocess OHLCV data (already handles weekends in the function)
    print("Preprocessing stock data...")
    stock_data, stock_names = preprocess_ohlcv_data("DataFromJanToApril")
    
    # 2. Create sector indices
    print("Creating sector indices...")
    sector_df = pd.read_csv(os.path.join(config.DATA_DIR, "sector_mapping.csv"))
    sectors = sector_df['sector'].tolist()
    sector_to_idx = {s: i for i, s in enumerate(set(sectors))}
    sector_indices = [sector_to_idx[s] for s in sectors]
    
    # 3. Create data loader
    print("Creating data loader...")
    data_loader = create_data_loader(stock_data, stock_names, sector_indices, batch_size=1)
    
    # 4. Load the model
    print("Loading model...")
    checkpoint = torch.load(os.path.join(config.CHECKPOINTS_DIR, "best_model_dim8.pt"), 
                          map_location=device, weights_only=False)
    
    # 5. Initialize model with loaded parameters
    model = FinGAT(
        input_dim=16,  # Number of features per stock
        hidden_dim=checkpoint['hidden_dim'],
        embed_dim=config.EMBEDDING_DIM,
        num_stocks=len(stock_names),
        num_sectors=19,
        num_weeks=config.NUM_WEEKS,
        transformer_layers=config.TRANSFORMER_LAYERS,
        transformer_heads=config.TRANSFORMER_HEADS,
        dropout=0.1,
        delta=config.DELTA
    ).to(device)
    
    # 6. Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 7. Generate daily predictions and metrics
    print("Generating predictions...")
    metrics, rankings = generate_daily_predictions(model, data_loader, device, stock_names, all_dates)
    
    # 8. Print summary (only includes business days)
    if metrics:
        # print("\nInstance Metrics Summary (Business Days Only):")
        print(" Test Result(From 11 Jan to 22 March):")
        metrics_df = pd.DataFrame(metrics)
        for k in [5, 10, 20]:
            k_metrics = metrics_df[metrics_df['K'] == k]
            print(f"K={k}:")
            print(f"   MRR: {k_metrics['MRR'].mean():.6f}")
            print(f"  Precision: {k_metrics['Precision'].mean():.6f}")
            print(f"   IRR: {k_metrics['IRR'].mean():.6f}")
    
    print("\nProcessing complete. Results saved to:", config.RESULTS_DIR)

if __name__ == "__main__":
    main()