# utils/preprocessing.py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch
import config

def add_features(df):
    """Add financial features to the dataframe as specified in Task2"""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure the dataframe is sorted by date
    df = df.sort_values('Date')
    
    # Calculate return ratio
    df['return_ratio'] = df['Close'].pct_change()
    
    # Calculate percentage change in open, high, low
    df['pct_change_open'] = df['Open'].pct_change()
    df['pct_change_high'] = df['High'].pct_change()
    df['pct_change_low'] = df['Low'].pct_change()
    
    # Calculate moving averages for 5, 10, 15, 20, 25, 30 days
    for window in [5, 10, 15, 20, 25, 30]:
        df[f'ma_{window}'] = df['Close'].rolling(window=window).mean()
    
    # Calculate binary label (1 if price went up, 0 if down)
    df['up_or_down'] = (df['Close'].diff() > 0).astype(int)
    
    # Fill NaN values with 0
    df = df.fillna(0)
    
    return df

def create_sliding_windows(df, window_size=5, prediction_horizon=1):
    """
    Create sliding windows of data with window_size days and labels 
    for prediction_horizon days ahead (sliding window approach from Task2)
    """
    data = []
    labels = []
    
    # Ensure the dataframe is sorted by date
    df = df.sort_values('Date')
    
    # Feature columns (all except Date and labels)
    feature_cols = [col for col in df.columns if col not in ['Date', 'Ticker','Sector']]
    
    # Create sliding windows
    for i in range(len(df) - window_size - prediction_horizon + 1):
        # Get window of data (5 days)
        window = df.iloc[i:i+window_size][feature_cols].values
        
        # Get label (return ratio for the 6th day after window)
        label_return = df.iloc[i+window_size+prediction_horizon-1]['return_ratio']
        label_movement = df.iloc[i+window_size+prediction_horizon-1]['up_or_down']
        
        data.append(window)
        labels.append((label_return, label_movement))
    
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)

def split_data_by_days(data, train_days=400, val_days=160):
    """Split data into train, validation, and test sets based on days"""
    train_data = data[:train_days]
    val_data = data[train_days:train_days+val_days]
    test_data = data[train_days+val_days:]
    
    return train_data, val_data, test_data

def process_all_stocks():
    """Process all 455 stocks in the cleaned directory"""
    # Get all CSV files in the cleaned directory
    csv_files = [f for f in os.listdir(config.CLEANED_DIR) if f.endswith('.csv')]
    
    all_processed_data = []
    all_stock_names = []
    all_sectors = []
    
    # Process each stock file
    for csv_file in tqdm(csv_files, desc="Processing stocks"):
        # Read CSV file
        df = pd.read_csv(os.path.join(config.CLEANED_DIR, csv_file))
        
        # Get stock name from filename
        stock_name = csv_file.split('.')[0]
        
        # Extract sector from the dataframe
        if 'Sector' not in df.columns:
            raise ValueError(f"CSV file {csv_file} is missing required 'Sector' column")
        sector = df['Sector'].iloc[0]  # Get sector from first row
        
        # Add features to the stock data
        df = add_features(df)
        
        # Save processed data
        df.to_csv(os.path.join(config.PROCESSED_DIR, csv_file), index=False)
        
        # Create sliding windows
        data, labels = create_sliding_windows(df, window_size=config.WINDOW_SIZE)
        
        # Save windowed data
        np.save(os.path.join(config.WINDOWS_DIR, f"{stock_name}_data.npy"), data.astype(np.float32))
        np.save(os.path.join(config.WINDOWS_DIR, f"{stock_name}_labels.npy"), labels.astype(np.float32))
        
        all_processed_data.append((data, labels))
        all_stock_names.append(stock_name)
        all_sectors.append(sector)
    
    # Create sector mapping
    sector_mapping = {stock: sector for stock, sector in zip(all_stock_names, all_sectors)}
    
    # Save sector mapping
    pd.DataFrame({'stock': all_stock_names, 'sector': all_sectors}).to_csv(
        os.path.join(config.DATA_DIR, "sector_mapping.csv"), index=False)
    
    # Create adjacency matrix based on sector
    num_stocks = len(all_stock_names)
    adj_matrix = np.zeros((num_stocks, num_stocks))
    
    for i in range(num_stocks):
        for j in range(num_stocks):
            if i!= j and all_sectors[i] == all_sectors[j]:
                adj_matrix[i, j] = 1
    
    # Save adjacency matrix
    np.save(os.path.join(config.DATA_DIR, "adj_matrix.npy"), adj_matrix)
    
    return all_processed_data, all_stock_names, sector_mapping, adj_matrix
