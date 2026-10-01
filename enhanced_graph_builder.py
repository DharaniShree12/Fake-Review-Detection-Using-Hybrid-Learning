"""Enhanced Graph Builder for User-Product Interactions"""
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

class EnhancedGraphBuilder:
    def __init__(self, add_similarity_edges=True, similarity_threshold=0.4):
        self.add_similarity_edges = add_similarity_edges
        self.similarity_threshold = similarity_threshold
    
    def build_enhanced_graph(self, df):
        """Build user-product bipartite graph"""
        print(f"Building graph from {len(df)} reviews...")
        
        # Create node mappings
        unique_users = df['UserId'].unique()
        unique_products = df['ProductId'].unique()
        
        user_to_idx = {user: i for i, user in enumerate(unique_users)}
        product_to_idx = {prod: i + len(unique_users) for i, prod in enumerate(unique_products)}
        
        # Create edges (user -> product)
        edge_index = []
        for _, row in df.iterrows():
            user_idx = user_to_idx[row['UserId']]
            product_idx = product_to_idx[row['ProductId']]
            edge_index.append([user_idx, product_idx])
            edge_index.append([product_idx, user_idx])  # Bidirectional
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Create node features with fixed dimension (128)
        num_nodes = len(unique_users) + len(unique_products)
        feature_dim = 128
        x = torch.randn(num_nodes, feature_dim) * 0.1  # Small random features
        
        graph_data = Data(x=x, edge_index=edge_index)
        
        print(f"Graph created: {num_nodes} nodes, {edge_index.shape[1]} edges")
        print(f"  Users: {len(unique_users)}, Products: {len(unique_products)}")
        
        return graph_data
    
    def extract_behavioral_features(self, df):
        """Extract behavioral features for each review"""
        print("Extracting behavioral features...")
        
        features = []
        
        # Group by user and product for statistics
        user_stats = df.groupby('UserId').agg({
            'Score': ['mean', 'std', 'count'],
            'label': 'mean'
        }).fillna(0)
        
        product_stats = df.groupby('ProductId').agg({
            'Score': ['mean', 'count'],
            'label': 'mean'
        }).fillna(0)
        
        for _, row in df.iterrows():
            user_id = row['UserId']
            product_id = row['ProductId']
            
            # User features
            user_avg_rating = user_stats.loc[user_id, ('Score', 'mean')]
            user_rating_std = user_stats.loc[user_id, ('Score', 'std')]
            user_review_count = user_stats.loc[user_id, ('Score', 'count')]
            user_fake_ratio = user_stats.loc[user_id, ('label', 'mean')]
            
            # Product features
            product_avg_rating = product_stats.loc[product_id, ('Score', 'mean')]
            product_review_count = product_stats.loc[product_id, ('Score', 'count')]
            product_fake_ratio = product_stats.loc[product_id, ('label', 'mean')]
            
            # Current review features
            rating_deviation = abs(row['Score'] - product_avg_rating)
            
            feature_vector = [
                user_avg_rating / 5.0,  # Normalize
                user_rating_std / 2.0,  # Normalize
                min(user_review_count / 10.0, 1.0),  # Cap at 10
                user_fake_ratio,
                product_avg_rating / 5.0,
                min(product_review_count / 50.0, 1.0),  # Cap at 50
                product_fake_ratio,
                rating_deviation / 5.0
            ]
            
            features.append(feature_vector)
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        print(f"Behavioral features shape: {features_tensor.shape}")
        
        return features_tensor
    
    def analyze_graph_patterns(self, df):
        """Analyze graph patterns for suspicious behavior"""
        print("Analyzing graph patterns...")
        
        # Calculate graph density
        num_users = df['UserId'].nunique()
        num_products = df['ProductId'].nunique()
        num_edges = len(df)
        max_edges = num_users * num_products
        graph_density = num_edges / max_edges if max_edges > 0 else 0
        
        # Reviews per user
        reviews_per_user = df.groupby('UserId').size()
        avg_reviews_per_user = reviews_per_user.mean()
        
        # Identify suspicious users (high fake ratio)
        user_fake_ratio = df.groupby('UserId')['label'].mean()
        suspicious_users = user_fake_ratio[user_fake_ratio > 0.7].index.tolist()
        
        analytics = {
            'graph_density': graph_density,
            'avg_reviews_per_user': avg_reviews_per_user,
            'suspicious_users': suspicious_users,
            'num_users': num_users,
            'num_products': num_products
        }
        
        return analytics


if __name__ == "__main__":
    print("Enhanced Graph Builder module loaded successfully")
