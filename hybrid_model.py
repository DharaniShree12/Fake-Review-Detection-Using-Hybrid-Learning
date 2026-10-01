"""Module 7 & 8: Feature Fusion and Classification"""
import torch
import torch.nn as nn
from attention_module import AttentionLayer
from config_amazonDS import HIDDEN_DIM, DROPOUT, NUM_CLASSES

class HybridFakeReviewDetector(nn.Module):
    def __init__(self, text_dim=768, behavioral_dim=8, gnn_dim=32, attention_dim=128, hidden_dim=HIDDEN_DIM):
        super(HybridFakeReviewDetector, self).__init__()
        
        # Attention layer for text features
        self.attention = AttentionLayer(input_dim=text_dim, attention_dim=attention_dim)
        
        # Feature fusion with improved architecture and residual connections
        fusion_input_dim = attention_dim + behavioral_dim
        
        # First fusion block
        self.fusion_block1 = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )
        
        # Second fusion block with residual
        self.fusion_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )
        
        # Third fusion block with residual
        self.fusion_block3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT * 0.8)
        )
        
        # Enhanced classification head with deeper architecture
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(DROPOUT * 0.7),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(DROPOUT * 0.5),
            nn.Linear(hidden_dim // 8, NUM_CLASSES)
        )
        
    def forward(self, text_embeddings, behavioral_features):
        """
        text_embeddings: (batch_size, text_dim)
        behavioral_features: (batch_size, behavioral_dim)
        """
        # Apply attention to text
        attended_text, attention_weights = self.attention(text_embeddings)
        
        # Concatenate features
        fused_features = torch.cat([attended_text, behavioral_features], dim=1)
        
        # Multi-block fusion with residual connections
        x = self.fusion_block1(fused_features)
        x = x + self.fusion_block2(x)  # Residual connection
        x = x + self.fusion_block3(x)  # Residual connection
        
        # Classification
        logits = self.classifier(x)
        
        return logits, attention_weights

class ModelTrainer:
    def __init__(self, model, device, class_weights=None):
        self.model = model.to(device)
        self.device = device
        self.warmup_steps = 0
        self.current_step = 0
        
        # Use weighted loss if class weights provided
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
            print(f"Using weighted loss with weights: {class_weights}")
            print(f"Label smoothing: 0.1 for better generalization")
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Use AdamW with weight decay
        from config_amazonDS import WEIGHT_DECAY, LEARNING_RATE
        self.base_lr = LEARNING_RATE
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.base_lr, 
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,  # Restart every 5 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-7
        )
        
    def get_lr_with_warmup(self, step, warmup_steps):
        """Linear warmup for the first warmup_steps"""
        if step < warmup_steps:
            return self.base_lr * (step / warmup_steps)
        return self.base_lr
    
    def train_epoch(self, text_emb, behavioral_feat, labels, batch_size=16, epoch=0):
        """Train for one epoch with warmup"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Set warmup steps for first 3 epochs
        if epoch < 3:
            self.warmup_steps = len(labels) // batch_size
        else:
            self.warmup_steps = 0
        
        indices = torch.randperm(len(labels))
        
        for i in range(0, len(labels), batch_size):
            batch_idx = indices[i:i+batch_size]
            
            batch_text = text_emb[batch_idx].to(self.device)
            batch_behavioral = behavioral_feat[batch_idx].to(self.device)
            batch_labels = labels[batch_idx].to(self.device)
            
            # Apply warmup learning rate
            if self.warmup_steps > 0:
                lr = self.get_lr_with_warmup(self.current_step % self.warmup_steps, self.warmup_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            self.optimizer.zero_grad()
            
            logits, _ = self.model(batch_text, batch_behavioral)
            loss = self.criterion(logits, batch_labels)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
            
            self.current_step += 1
        
        # Step scheduler after warmup
        if self.warmup_steps == 0:
            self.scheduler.step()
        
        avg_loss = total_loss / (len(labels) // batch_size)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, text_emb, behavioral_feat, labels):
        """Evaluate model"""
        self.model.eval()
        
        with torch.no_grad():
            text_emb = text_emb.to(self.device)
            behavioral_feat = behavioral_feat.to(self.device)
            labels = labels.to(self.device)
            
            logits, attention_weights = self.model(text_emb, behavioral_feat)
            
            _, predicted = torch.max(logits, 1)
            accuracy = (predicted == labels).sum().item() / labels.size(0)
        
        return accuracy, predicted.cpu(), attention_weights.cpu()
