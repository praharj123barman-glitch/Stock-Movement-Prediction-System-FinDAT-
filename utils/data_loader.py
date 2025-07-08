# utils/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import config
import pandas as pd
class StockDataset(Dataset):
    """Dataset for stock data with sliding windows"""
    
    def __init__(self, stock_data, stock_labels, stock_indices, sector_indices, window_indices):
        """
        stock_data: List of arrays containing stock features for each window
        stock_labels: List of arrays containing stock labels for each window
        stock_indices: Indices of stocks to include
        sector_indices: Sector index for each stock
        window_indices: Indices of windows to include (for train/val/test split)
        """
        self.stock_data = [stock_data[i] for i in stock_indices]
        self.stock_labels = [stock_labels[i] for i in stock_indices]
        self.stock_indices = stock_indices
        self.sector_indices = sector_indices
        self.window_indices = window_indices
        self.num_stocks = len(stock_indices)
    
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        window_idx = self.window_indices[idx]
        
        # Get data and labels for all stocks for this window
        batch_features = []
        batch_return_labels = []
        batch_movement_labels = []
        
        for i in range(self.num_stocks):
            if window_idx < len(self.stock_data[i]):
                features = self.stock_data[i][window_idx].astype(np.float32)
                return_label, movement_label = self.stock_labels[i][window_idx]
            else:
                # Handle case where a stock doesn't have data for this window
                features = np.zeros_like(self.stock_data[i][0],dtype=np.float32)
                return_label, movement_label = 0, 0
                
            batch_features.append(features)
            batch_return_labels.append(return_label)
            batch_movement_labels.append(movement_label)
        
        # Create adjacency matrix based on sector
        adj_matrix = np.zeros((self.num_stocks, self.num_stocks),dtype=np.float32)
        for i in range(self.num_stocks):
            for j in range(self.num_stocks):
                if i != j and self.sector_indices[i] == self.sector_indices[j]:
                    adj_matrix[i, j] = 1.0
        
        # Convert to tensors
        features_tensor = torch.as_tensor(np.array(batch_features),dtype=torch.float32)
        return_labels_tensor = torch.as_tensor(np.array(batch_return_labels),dtype=torch.float32)
        movement_labels_tensor = torch.as_tensor(np.array(batch_movement_labels),dtype=torch.float32)
        adj_matrix_tensor = torch.as_tensor(adj_matrix,dtype=torch.float32)
        sector_indices_tensor = torch.as_tensor(self.sector_indices,dtype=torch.long)
        
        return (features_tensor, adj_matrix_tensor, sector_indices_tensor, 
                return_labels_tensor, movement_labels_tensor)

def create_dataloaders(batch_size=32):
    """Create dataloaders for train, validation, and test sets"""
    # Load processed data
    stock_data = []
    stock_labels = []
    stock_names = []
    
    # Get all processed data files
    data_files = [f for f in os.listdir(config.WINDOWS_DIR) if f.endswith('_data.npy')]
    
    for data_file in data_files:
        stock_name = data_file.split('_')[0]
        
        # Load data and labels
        data = np.load(os.path.join(config.WINDOWS_DIR, data_file),allow_pickle=True)
        labels = np.load(os.path.join(config.WINDOWS_DIR, f"{stock_name}_labels.npy"),allow_pickle=True)
        
        stock_data.append(data.astype(np.float32))
        stock_labels.append(labels.astype(np.float32))
        stock_names.append(stock_name)
        
    # Load sector mapping
    sector_mapping_df = pd.read_csv(os.path.join(config.DATA_DIR, "sector_mapping.csv"))
    
    # Convert sectors to numerical indices
    sectors = list(sector_mapping_df['sector'].unique())
    sector_to_idx = {sector: i for i, sector in enumerate(sectors)}
    
    # Get sector indices for each stock
    sector_indices = []
    for stock in stock_names:
        sector = sector_mapping_df[sector_mapping_df['stock'] == stock]['sector'].values[0]
        sector_indices.append(sector_to_idx[sector])
    
    # Calculate window indices for train/val/test split
    min_windows = min(len(data) for data in stock_data)
    
    # Ensure we have enough data for the split
    if min_windows < config.TRAIN_DAYS + config.VAL_DAYS:
        raise ValueError("Not enough data for train/val split")
        
    train_indices = list(range(config.TRAIN_DAYS))
    val_indices = list(range(config.TRAIN_DAYS, config.TRAIN_DAYS + config.VAL_DAYS))
    test_indices = list(range(config.TRAIN_DAYS + config.VAL_DAYS, min_windows))
    
    # Create stock indices (all stocks)
    stock_indices = list(range(len(stock_names)))
    
    # Create datasets
    train_dataset = StockDataset(stock_data, stock_labels, stock_indices, 
                                sector_indices, train_indices)
    val_dataset = StockDataset(stock_data, stock_labels, stock_indices,
                              sector_indices, val_indices)
    test_dataset = StockDataset(stock_data, stock_labels, stock_indices,
                               sector_indices, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
