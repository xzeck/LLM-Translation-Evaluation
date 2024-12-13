from transformers import AutoTokenizer, AutoModel
import torch

# Embedding Model Class
class EmbeddingModel:
    def __init__(self, model_name="thenlper/gte-small", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length  # Maximum sequence length for truncation

    def get_embeddings(self, text_list):
        # Tokenize with truncation and padding
        inputs = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,  # Ensure sequences are truncated to fit the model's input
            max_length=self.max_length,  # Limit sequence length
            return_tensors="pt"
        )
        # Compute embeddings without gradient calculation
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Average pooling over token embeddings for sequence representation
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.numpy()
