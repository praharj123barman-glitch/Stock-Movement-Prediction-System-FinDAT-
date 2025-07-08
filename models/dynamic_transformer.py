
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicAttention(nn.Module):
    """
    Dynamic Attention mechanism for graph-structured data
    Replaces the standard GAT with a more expressive transformer attention
    """
    def __init__(self, embed_dim, num_heads=4):
        super(DynamicAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Make sure embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        # Linear projections for queries, keys, values
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        # Scaling factor for dot-product attention
        self.scaling = self.head_dim ** -0.5
    
    def forward(self, x, adj_matrix):
        """
        x: Node features of shape (batch_size, num_nodes, embed_dim)
        adj_matrix: Adjacency matrix of shape (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = x.size()
        
        # Project queries, keys, values
        # Shape: (batch_size, num_heads, num_nodes, head_dim)
        queries = self.query_projection(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2) #	What I’m looking for
        keys = self.key_projection(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)  #	What I offer
        values = self.value_projection(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2) #What I carry

        
        # Calculate attention scores
        # scores: (batch_size, num_heads, num_nodes, num_nodes)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scaling  #Each score [i][j] shows how much Node i's Query matches Node j's Key.
        
        # Apply adjacency matrix as mask (optional, can be disabled for full attention)
        # This restricts attention to only connected nodes in the graph
        mask = adj_matrix.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        # attn_weights: (batch_size, num_heads, num_nodes, num_nodes)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        # context: (batch_size, num_heads, num_nodes, head_dim)
        context = torch.matmul(attn_weights, values)
        
        # Reshape and project output
        # output: (batch_size, num_nodes, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.embed_dim)
        output = self.output_projection(context)
        
        return output, attn_weights

class DynamicTransformerLayer(nn.Module):
    """Complete transformer layer with dynamic attention, normalization, and feed-forward"""
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(DynamicTransformerLayer, self).__init__()
        
        # Dynamic attention mechanism
        self.attention = DynamicAttention(embed_dim, num_heads)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj_matrix):
        """
        x: Node features of shape (batch_size, num_nodes, embed_dim)
        adj_matrix: Adjacency matrix of shape (batch_size, num_nodes, num_nodes)
        """
        # Apply dynamic attention with residual connection and normalization
        attention_output, _ = self.attention(x, adj_matrix)
        x = self.layer_norm1(x + self.dropout(attention_output)) 
        
        # Apply feed-forward network with residual connection and normalization
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x

class DynamicTransformer(nn.Module):
    """Multi-layer dynamic transformer for graph-structured data"""
    def __init__(self, embed_dim, num_layers=2, num_heads=4, dropout=0.1):
        super(DynamicTransformer, self).__init__()
        
        # Stack of transformer layers
        self.layers = nn.ModuleList([
            DynamicTransformerLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, adj_matrix):
        """
        x: Node features of shape (batch_size, num_nodes, embed_dim)
        adj_matrix: Adjacency matrix of shape (batch_size, num_nodes, num_nodes)
        """
        # Apply each transformer layer sequentially
        for layer in self.layers:
            x = layer(x, adj_matrix)
        
        return x
