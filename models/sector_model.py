# models/sector_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dynamic_transformer import DynamicTransformer

class SectorModel(nn.Module):
    """
    Sector-level modeling as described in Task 3
    Creates inter-sector graph and models relationships between sectors
    """
    def __init__(self, embed_dim, num_sectors, num_layers=2, num_heads=4, dropout=0.1):
        super(SectorModel, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_sectors = num_sectors
        
        # Dynamic transformer for inter-sector modeling
        self.transformer = DynamicTransformer(embed_dim, num_layers, num_heads, dropout)

    def forward(self, stock_embeddings, sector_indices):
        """
        stock_embeddings: Embeddings of stocks (batch_size, num_stocks, embed_dim)
        sector_indices: Sector index for each stock (batch_size, num_stocks)
        """
        batch_size, num_stocks, _ = stock_embeddings.size()
        
        # Initialize sector embeddings
        sector_embeddings = torch.zeros(batch_size, self.num_sectors, self.embed_dim).to(stock_embeddings.device)
        
        # Process each sample in the batch independently
        for batch_idx in range(batch_size):
            # Get sector indices for current batch
            current_sectors = sector_indices[batch_idx]  # Shape: [num_stocks]
            
            # Create sector embeddings through max pooling
            for sector_idx in range(self.num_sectors):
                # Get indices of stocks in this sector for current batch
                mask = (current_sectors == sector_idx)
                if mask.sum() > 0:
                    # Select embeddings for this sector
                    sector_stocks = stock_embeddings[batch_idx, mask]
                    # Max pool along the stock dimension
                    sector_embeddings[batch_idx, sector_idx] = torch.max(sector_stocks, dim=0)[0]

        # Create fully connected adjacency matrix for sectors (batch_size, num_sectors, num_sectors)
        sector_adj_matrix = torch.ones(batch_size, self.num_sectors, self.num_sectors).to(stock_embeddings.device)
        
        # Apply transformer for inter-sector modeling
        refined_sector_embeddings = self.transformer(sector_embeddings, sector_adj_matrix)
        
        # Map sector embeddings back to stocks
        refined_stock_embeddings = torch.zeros_like(stock_embeddings)
        for batch_idx in range(batch_size):
            current_sectors = sector_indices[batch_idx]  # Shape: [num_stocks]
            for stock_idx in range(num_stocks):
                sector_idx = current_sectors[stock_idx].item()
                refined_stock_embeddings[batch_idx, stock_idx] = refined_sector_embeddings[batch_idx, sector_idx]
        
        return refined_stock_embeddings
