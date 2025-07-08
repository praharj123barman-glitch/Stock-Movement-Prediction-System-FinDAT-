# models/attention_gru.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentiveGRU(nn.Module):
    """
    Attentive GRU for short-term sequential learning as described in Task 2
    Processes daily data and produces weekly embeddings using attention mechanism
    """
    def __init__(self, input_dim, hidden_dim):
        super(AttentiveGRU, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # GRU layer to process sequential data
        self.gru = nn.GRU(
            input_size=input_dim,  #no of feature per time step 
            hidden_size=hidden_dim,  # in this model I use 8 (GRU_hiiden dim=8)
            batch_first=True  #: Expects input as (batch, sequence, features) instead of (sequence, batch, features)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  #Linear layer transforming hidden states
            nn.Tanh(),  #Tanh activation (squashes values to [-1, 1])
            nn.Linear(hidden_dim, 1)  #Final linear layer producing a single attention score
        )
    
    def forward(self, x):
        """
        x: Input sequence of shape (batch_size, seq_len, input_dim)
        Returns: Context vector of shape (batch_size, hidden_dim)
        """
        # Initialize hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)  #Shape: (1, batch_size, hidden_dim)
        
        # Run GRU
        # out: (batch_size, seq_len, hidden_dim)
        out, _ = self.gru(x, h0)  #Discards final hidden state (_) since we're using attention

        
        # Calculate attention weights
        # attn: (batch_size, seq_len, 1)
        attn = self.attention(out) #Passes GRU outputs through attention network
        
        # Apply softmax to get normalized weights
        # alpha: (batch_size, seq_len, 1)
        alpha = F.softmax(attn, dim=1)
        
        # Apply attention weights to get context vector
        # context: (batch_size, hidden_dim)
        context = torch.sum(out * alpha, dim=1)
        
        return context
