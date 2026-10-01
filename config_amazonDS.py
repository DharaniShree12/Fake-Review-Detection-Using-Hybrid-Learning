"""Configuration file for AmazonDS Labeled Dataset Training - Optimized for 15K records"""

# Data Configuration
DATA_PATH = "AmazonDS/train.csv"
NUM_REVIEWS = 15000  # Increased from 10000
TRAIN_SPLIT = 0.85
RANDOM_SEED = 42

# Model Configuration - Using DistilBERT with optimized parameters for 15K
TRANSFORMER_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 6e-6  # Lower for stability with more data
NUM_EPOCHS = 40  # More epochs for larger dataset
WEIGHT_DECAY = 0.025  # Stronger regularization for more data
EARLY_STOPPING_PATIENCE = 12

# GNN Configuration - Enhanced for 15K
GNN_HIDDEN_DIM = 384  # Increased capacity
GNN_OUTPUT_DIM = 192  # Increased capacity
GNN_NUM_LAYERS = 3

# Attention Configuration - Enhanced
ATTENTION_HEADS = 12
ATTENTION_DIM = 512  # Increased from 480

# Classification Configuration - Enhanced
HIDDEN_DIM = 1408  # Increased from 1280
DROPOUT = 0.15  # Optimal for 15K samples
NUM_CLASSES = 2
USE_WEIGHTED_LOSS = True

# Visualization Configuration
FIGSIZE = (10, 6)
