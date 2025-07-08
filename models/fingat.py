import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention_gru import AttentiveGRU
from models.dynamic_transformer import DynamicTransformer
from models.sector_model import SectorModel

class FinGAT(nn.Module):
    """
    Financial Graph Attention Network with Dynamic Transformer
    Complete model integrating short-term sequential learning, intra-sector modeling,
    long-term sequential learning, and sector-level modeling
    """
    def __init__(self, input_dim, hidden_dim, embed_dim, num_stocks, num_sectors,
                 num_weeks=4, transformer_layers=2, transformer_heads=4, dropout=0.1, delta=0.5):
        super(FinGAT, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim #this is used in gru
        self.embed_dim = embed_dim    #this is used in dynamic transformer model
        self.num_stocks = num_stocks   #455
        self.num_sectors = num_sectors  #19
        self.delta = delta
        
        # Short-term sequential learning (Task 2)
        self.short_term_gru = AttentiveGRU(input_dim, hidden_dim)
          # Projection from hidden_dim to embed_dim
        self.stock_proj = nn.Linear(hidden_dim, embed_dim)
        
        # Intra-sector relation modeling with Dynamic Transformer (replacing GAT)
        self.intra_sector_transformer = DynamicTransformer(
            embed_dim, transformer_layers, transformer_heads, dropout
        )
        
        # Long-term sequential learning with GRU
        self.long_term_gru = AttentiveGRU(embed_dim, hidden_dim)
        self.long_term_proj = nn.Linear(hidden_dim, embed_dim)
        
        # Sector-level modeling (Task 3)
        self.sector_model = SectorModel(
            embed_dim, num_sectors, transformer_layers, transformer_heads, dropout
        )
        
        # Embedding fusion
        self.fusion = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific prediction layers
        self.return_predictor = nn.Linear(embed_dim, 1)
        self.movement_predictor = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, stock_features, adj_matrix, sector_indices, historical_embeddings=None):
        """
        stock_features: Features of stocks (batch_size, num_stocks, seq_len, input_dim)
        adj_matrix: Adjacency matrix (batch_size, num_stocks, num_stocks)
        sector_indices: Sector index for each stock (num_stocks)
        historical_embeddings: Optional historical embeddings for long-term learning
        """
        batch_size, num_stocks, seq_len, _ = stock_features.size()
        
        # Short-term sequential learning for each stock
        stock_embeddings = torch.zeros(batch_size, num_stocks, self.embed_dim).to(stock_features.device)
        
        for i in range(num_stocks):
            # Process each stock's time series with GRU
            stock_hidden = self.short_term_gru(stock_features[:, i])
            stock_embeddings[:, i] = self.stock_proj(stock_hidden)
        
        # Intra-sector relation modeling with dynamic transformer
        intra_sector_embeddings = self.intra_sector_transformer(stock_embeddings, adj_matrix)
        
        # Long-term sequential learning (if historical embeddings provided)
        if historical_embeddings is not None:
            long_term_embeddings = torch.zeros(batch_size, num_stocks, self.embed_dim).to(stock_features.device)
            
            for i in range(num_stocks):
                # Process historical embeddings with GRU
                long_term_hidden = self.long_term_gru(historical_embeddings[:, i])
                long_term_embeddings[:, i] = self.long_term_proj(long_term_hidden)
        else:
            # If no historical data, initialize with zeros
            long_term_embeddings = torch.zeros(batch_size, num_stocks, self.embed_dim).to(stock_features.device)
        
        # Sector-level modeling
        sector_embeddings = self.sector_model(intra_sector_embeddings, sector_indices)
        
        # Embedding fusion
        fused_embeddings = self.fusion(
            torch.cat([intra_sector_embeddings, long_term_embeddings, sector_embeddings], dim=2)
        )
        
        # Task-specific predictions
        return_predictions = self.return_predictor(fused_embeddings).squeeze(-1)
        movement_predictions = self.movement_predictor(fused_embeddings).squeeze(-1)
        
        return return_predictions, movement_predictions
    
    def calculate_loss(self, return_preds, movement_preds, return_labels, movement_labels):
        """
        Calculate multi-task loss combining ranking-aware loss and movement prediction loss
        as described in the FinGAT paper
        """
        # Ranking loss (pairwise comparison)
        ranking_loss = 0.0
        batch_size, num_stocks = return_preds.size()
        
        for i in range(batch_size):
            for j in range(num_stocks):
                for k in range(j+1, num_stocks):
                    # Get predictions and true values
                    pred_diff = return_preds[i, j] - return_preds[i, k]
                    true_diff = return_labels[i, j] - return_labels[i, k]
                    
                    # Calculate hinge loss for ranking
                    if true_diff * pred_diff < 0:  # Wrong ordering
                        ranking_loss += torch.abs(true_diff)
        
        # Movement prediction loss (binary cross-entropy)
        movement_loss = F.binary_cross_entropy(
            movement_preds, movement_labels.float()
        )
        
        # Combine losses with delta parameter
        total_loss = (1 - self.delta) * ranking_loss + self.delta * movement_loss
        
        return total_loss
