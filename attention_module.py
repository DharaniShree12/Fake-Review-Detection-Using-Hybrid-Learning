"""Module 4: Attention Mechanism for Interpretability"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config_amazonDS import ATTENTION_HEADS, ATTENTION_DIM

class AttentionLayer(nn.Module):
    def __init__(self, input_dim=768, attention_dim=ATTENTION_DIM, num_heads=ATTENTION_HEADS):
        super(AttentionLayer, self).__init__()
        
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        
        # Multi-head attention
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        
        self.fc_out = nn.Linear(attention_dim, attention_dim)
        
    def forward(self, x):
        """
        x: (batch_size, embedding_dim)
        Returns: attended features and attention weights
        """
        batch_size = x.size(0)
        
        # Expand for attention computation
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        
        # Compute Q, K, V
        Q = self.query(x_expanded)
        K = self.key(x_expanded)
        V = self.value(x_expanded)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention_scores = attention_scores / (self.attention_dim ** 0.5)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.squeeze(1)
        
        # Final linear layer
        output = self.fc_out(attended)
        
        return output, attention_weights.squeeze(1)
