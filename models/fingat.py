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
    
    def encode_short(self, x):
        """Shared short-term encoder used for BOTH the current window and the
        historical weekly windows (Stage 3).

        x: (num_sequences, seq_len, input_dim) -> (num_sequences, embed_dim)
        """
        return self.stock_proj(self.short_term_gru(x))

    def forward(self, stock_features, adj_matrix, sector_indices, historical_windows=None):
        """
        stock_features: Features of stocks (batch_size, num_stocks, seq_len, input_dim)
        adj_matrix: Adjacency matrix (batch_size, num_stocks, num_stocks)
        sector_indices: Sector index for each stock (batch_size, num_stocks)
        historical_windows: Optional raw historical windows for long-term (Stage 3)
            learning, shape (batch_size, num_stocks, num_weeks, seq_len, input_dim).
            When provided, the long-term Attentive GRU branch is ACTIVE; when None
            it falls back to zeros (backward-compatible with old 3-argument callers).
        """
        batch_size, num_stocks, seq_len, input_dim = stock_features.size()

        # Short-term sequential learning (vectorized: one GRU call for all stocks
        # instead of a Python loop over ~445 stocks -> ~5x faster).
        flat = stock_features.reshape(batch_size * num_stocks, seq_len, input_dim)
        stock_embeddings = self.encode_short(flat).reshape(batch_size, num_stocks, self.embed_dim)

        # Intra-sector relation modeling with dynamic transformer
        intra_sector_embeddings = self.intra_sector_transformer(stock_embeddings, adj_matrix)

        # Long-term sequential learning (Stage 3) -- ACTIVE when history is supplied.
        # Encode each of the last `num_weeks` windows with the SAME short-term
        # encoder, then run the long-term Attentive GRU over that week-sequence.
        if historical_windows is not None:
            num_weeks = historical_windows.size(2)
            hw = historical_windows.reshape(batch_size * num_stocks * num_weeks, seq_len, input_dim)
            weekly_embeddings = self.encode_short(hw).reshape(batch_size * num_stocks, num_weeks, self.embed_dim)
            long_term_hidden = self.long_term_gru(weekly_embeddings)          # (B*N, hidden_dim)
            long_term_embeddings = self.long_term_proj(long_term_hidden).reshape(
                batch_size, num_stocks, self.embed_dim)                       # real vector, not zeros
        else:
            # No history provided -> keep the old zero behaviour (branch inactive).
            long_term_embeddings = torch.zeros(
                batch_size, num_stocks, self.embed_dim, device=stock_features.device)

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

    def ranking_movement_loss(self, return_preds, movement_preds,
                              return_labels, movement_labels, margin=0.05):
        """Corrected, vectorized multi-task loss.

        The original `calculate_loss` above adds `abs(true_diff)` on mis-ordered
        pairs. That term does NOT depend on the predictions, so it contributes a
        ZERO gradient to the return head and the ranking predictions collapse to a
        near-constant. This version instead applies a differentiable margin hinge
        on the *predicted* pairwise differences, and rescales the ranking term by
        (batch_size * num_stocks) so it is not drowned out by the movement BCE.
        Fully tensorized -> one GPU op instead of a triple Python loop.
        """
        # Pairwise differences: (batch, num_stocks, num_stocks)
        pred_diff = return_preds.unsqueeze(2) - return_preds.unsqueeze(1)
        true_diff = return_labels.unsqueeze(2) - return_labels.unsqueeze(1)

        # Hinge: penalise when the predicted order disagrees with the true order,
        # weighted by how large the true gap is.
        pair = F.relu(margin - torch.sign(true_diff) * pred_diff) * true_diff.abs()

        batch_size, num_stocks = return_preds.size()
        ranking = pair.sum() / (batch_size * num_stocks)
        movement = F.binary_cross_entropy(movement_preds, movement_labels.float())

        return (1 - self.delta) * ranking + self.delta * movement
