"""Module 6: Graph Neural Network for Behavioral Learning"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from config_amazonDS import GNN_HIDDEN_DIM, GNN_OUTPUT_DIM, GNN_NUM_LAYERS

class GNNEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=GNN_HIDDEN_DIM, output_dim=GNN_OUTPUT_DIM, num_layers=GNN_NUM_LAYERS):
        super(GNNEncoder, self).__init__()
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index):
        """
        x: Node features
        edge_index: Graph connectivity
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.relu(x)
                x = self.dropout(x)
        
        return x

class GNNFeatureExtractor:
    def __init__(self, graph_data):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.graph_data = graph_data.to(self.device)
        self.model = GNNEncoder().to(self.device)
        
    def extract_embeddings(self):
        """Extract GNN embeddings for all nodes"""
        self.model.eval()
        
        with torch.no_grad():
            embeddings = self.model(self.graph_data.x, self.graph_data.edge_index)
        
        return embeddings.cpu()
