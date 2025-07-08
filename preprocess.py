
# preprocess.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import config
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))


from utils.preprocessing import add_features, create_sliding_windows, process_all_stocks

def main():
    """Preprocess all stock data files"""
    print("Starting data preprocessing...")
    
    # Create directories if they don't exist
    for directory in [config.PROCESSED_DIR, config.WINDOWS_DIR, config.SPLITS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Process all stocks
    all_processed_data, all_stock_names, sector_mapping, adj_matrix = process_all_stocks()
    
    print(f"Processed {len(all_stock_names)} stocks with features:")
    for feature in config.FEATURES:
        print(f"  - {feature}")
    
    print(f"Created sector mapping with {len(set(sector_mapping.values()))} sectors")
    print(f"Data split: {config.TRAIN_DAYS} training days, {config.VAL_DAYS} validation days, remaining for testing")
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()
