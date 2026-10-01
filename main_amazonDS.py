"""Main Pipeline for AmazonDS Labeled Dataset with DistilBERT"""
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from amazonDS_data_loader import AmazonDSDataLoader
from preprocessor import TextPreprocessor
from transformer_encoder import TransformerEncoder
from enhanced_graph_builder import EnhancedGraphBuilder
from gnn_module import GNNFeatureExtractor
from hybrid_model import HybridFakeReviewDetector, ModelTrainer
from visualizer import ModelVisualizer
import config_amazonDS as config

def main():
    print("="*60)
    print("AMAZONDS LABELED DATASET - DISTILBERT HYBRID MODEL")
    print("Target: >90% Accuracy with Real Labels")
    print("="*60)
    
    # Module 1: Data Loading
    print("\n[1/10] Loading AmazonDS Labeled Data...")
    loader = AmazonDSDataLoader(
        data_path=config.DATA_PATH,
        num_reviews=config.NUM_REVIEWS
    )
    df = loader.load_data()
    df = loader.create_synthetic_labels(df)  # Just prints message, labels already exist
    
    # Module 2: Preprocessing
    print("\n[2/10] Preprocessing Data...")
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df)
    
    # Module 3: Transformer Encoding
    print("\n[3/10] Extracting Text Features with DistilBERT...")
    encoder = TransformerEncoder()
    text_embeddings, _ = encoder.encode_texts(df['cleaned_text'].tolist())
    
    # Module 4: Attention (integrated in model)
    print("\n[4/10] Attention mechanism ready (integrated in model)...")
    
    # Module 5: Graph Construction
    print("\n[5/10] Building Enhanced Interaction Graph...")
    graph_builder = EnhancedGraphBuilder(
        add_similarity_edges=True,
        similarity_threshold=0.4  # Higher threshold to reduce edges for 10K dataset
    )
    
    graph_data = graph_builder.build_enhanced_graph(df)
    behavioral_features = graph_builder.extract_behavioral_features(df)
    
    # Analyze graph patterns
    graph_analytics = graph_builder.analyze_graph_patterns(df)
    print(f"\nGraph Analytics:")
    print(f"  - Graph density: {graph_analytics['graph_density']:.4f}")
    print(f"  - Avg reviews/user: {graph_analytics['avg_reviews_per_user']:.2f}")
    print(f"  - Suspicious users detected: {len(graph_analytics['suspicious_users'])}")
    
    # Module 6: GNN Feature Extraction
    print("\n[6/10] Extracting GNN Features...")
    gnn_extractor = GNNFeatureExtractor(graph_data)
    gnn_embeddings = gnn_extractor.extract_embeddings()
    print(f"GNN embeddings shape: {gnn_embeddings.shape}")
    
    # Prepare labels
    labels = torch.tensor(df['label'].values, dtype=torch.long)
    
    # Split data
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, 
        train_size=config.TRAIN_SPLIT, 
        random_state=config.RANDOM_SEED, 
        stratify=labels
    )
    
    train_text = text_embeddings[train_idx]
    train_behavioral = behavioral_features[train_idx]
    train_labels = labels[train_idx]
    
    test_text = text_embeddings[test_idx]
    test_behavioral = behavioral_features[test_idx]
    test_labels = labels[test_idx]
    
    print(f"\nTrain size: {len(train_idx)}, Test size: {len(test_idx)}")
    
    # Module 7 & 8: Hybrid Model Training
    print("\n[7/10] Building Hybrid Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridFakeReviewDetector(
        text_dim=text_embeddings.shape[1],
        behavioral_dim=behavioral_features.shape[1],
        attention_dim=config.ATTENTION_DIM,
        hidden_dim=config.HIDDEN_DIM
    )
    
    print("\n[8/10] Training Model with REAL LABELS...")
    
    # Calculate class weights for imbalanced dataset
    if config.USE_WEIGHTED_LOSS:
        class_counts = torch.bincount(labels)
        total_samples = len(labels)
        class_weights = total_samples / (len(class_counts) * class_counts.float())
        print(f"Class distribution: Real={class_counts[0]}, Fake={class_counts[1]}")
        print(f"Class weights: Real={class_weights[0]:.3f}, Fake={class_weights[1]:.3f}")
    else:
        class_weights = None
    
    trainer = ModelTrainer(model, device, class_weights=class_weights)
    
    train_losses = []
    train_accs = []
    best_acc = 0
    patience_counter = 0
    
    print(f"Training with warmup and label smoothing")
    print(f"Target: >90% accuracy with 10,000 labeled reviews")
    
    for epoch in range(config.NUM_EPOCHS):
        loss, acc = trainer.train_epoch(
            train_text, train_behavioral, train_labels, epoch=epoch
        )
        train_losses.append(loss)
        train_accs.append(acc)
        
        # Show progress every epoch
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Loss: {loss:.4f}, Accuracy: {acc:.4f}", end="")
        
        # Early stopping check
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'amazonDS_best_model.pth')
            print(f" ← Best!")
        else:
            patience_counter += 1
            print()
            
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best training accuracy: {best_acc:.4f}")
            # Load best model
            model.load_state_dict(torch.load('amazonDS_best_model.pth'))
            break
    
    # Module 9: Evaluation & Visualization
    print("\n[9/10] Evaluating Model...")
    test_acc, test_pred, attention_weights = trainer.evaluate(
        test_text, test_behavioral, test_labels
    )
    print(f"Test Accuracy: {test_acc:.4f}")
    
    visualizer = ModelVisualizer()
    
    # Classification report
    visualizer.print_classification_report(test_labels.numpy(), test_pred.numpy())
    
    # Confusion matrix
    visualizer.plot_confusion_matrix(
        test_labels.numpy(), 
        test_pred.numpy(),
        save_path='amazonDS_confusion_matrix.png'
    )
    
    # ROC curve
    model.eval()
    with torch.no_grad():
        logits, _ = model(test_text.to(device), test_behavioral.to(device))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    visualizer.plot_roc_curve(
        test_labels.numpy(), 
        probs,
        save_path='amazonDS_roc_curve.png'
    )
    
    # Training history
    visualizer.plot_training_history(
        train_losses, 
        train_accs,
        save_path='amazonDS_training_history.png'
    )
    
    # Save model
    torch.save(model.state_dict(), 'amazonDS_fake_review_model.pth')
    print("\nModel saved to 'amazonDS_fake_review_model.pth'")
    
    # Save ALL predictions (train + test) for UI
    print("\nGenerating predictions for all reviews...")
    all_acc, all_pred, all_attention = trainer.evaluate(
        text_embeddings, behavioral_features, labels
    )
    
    df_all = df.copy()
    df_all['prediction'] = all_pred.numpy()
    
    # Get probabilities for all reviews
    model.eval()
    with torch.no_grad():
        logits, _ = model(
            text_embeddings.to(device), 
            behavioral_features.to(device)
        )
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    df_all['confidence'] = probs
    
    df_all.to_csv('amazonDS_predictions.csv', index=False)
    print(f"All {len(df_all)} predictions saved to 'amazonDS_predictions.csv'")
    print(f"  - Predicted Fake: {sum(all_pred.numpy())}")
    print(f"  - Predicted Real: {len(all_pred) - sum(all_pred.numpy())}")
    
    print("\n[10/10] Pipeline Complete!")
    print("="*60)
    print(f"\nFINAL RESULTS:")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Dataset: AmazonDS (REAL LABELS)")
    print(f"  Reviews: {config.NUM_REVIEWS}")
    if test_acc >= 0.90:
        print(f"  ✓ TARGET ACHIEVED: >90% accuracy!")
    else:
        print(f"  Target: 90%, Achieved: {test_acc*100:.2f}%")
    print("="*60)
    
    return model, df_all

if __name__ == "__main__":
    model, results_df = main()
