"""Module 3: Text Feature Extraction using Transformer"""
import torch
from transformers import AutoTokenizer, AutoModel
from config_amazonDS import TRANSFORMER_MODEL, MAX_LENGTH

class TransformerEncoder:
    def __init__(self, model_name=TRANSFORMER_MODEL):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def encode_texts(self, texts, max_length=MAX_LENGTH):
        """Convert texts to transformer embeddings"""
        print(f"Encoding {len(texts)} texts...")
        
        embeddings = []
        attention_masks = []
        
        with torch.no_grad():
            for i in range(0, len(texts), 16):
                batch_texts = texts[i:i+16]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Get embeddings
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Use [CLS] token embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
                
                embeddings.append(cls_embeddings)
                attention_masks.append(attention_mask.cpu())
        
        embeddings = torch.cat(embeddings, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        return embeddings, attention_masks
