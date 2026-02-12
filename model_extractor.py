import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


class ImageFeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True):
        super().__init__()
        if model_name == 'resnet18':
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            # Remove the final classification layer
            self.feature_dim = self.model.fc.in_features
            self.model.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        features = self.model(x)
        return features


class TextFeatureExtractor(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Get the hidden size for the model
        self.feature_dim = self.model.config.hidden_size
        
        # Freeze the base model parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the CLS token representation
        features = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        return features


def get_feature_extractor(data_type, model_name=None):
    """Get appropriate feature extractor based on data type"""
    if data_type == 'image':
        return ImageFeatureExtractor(model_name or 'resnet18')
    elif data_type == 'text':
        return TextFeatureExtractor(model_name or 'distilbert-base-uncased')
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def extract_embeddings(model, dataloader, device='cpu'):
    """Extract embeddings from the model for all data in the dataloader"""
    model.eval()
    all_embeddings = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                # Handle different data types
                if 'input_ids' in batch:
                    # Text data
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    embeddings = model(input_ids, attention_mask)
                    
                    # Get predictions (simple classification head)
                    logits = torch.randn(embeddings.size(0), 2)  # Placeholder for actual classifier
                    predictions = torch.argmax(logits, dim=1)
                else:
                    # Image data
                    images = batch['image'].to(device)
                    embeddings = model(images)
                    
                    # Get predictions (simple classification head)
                    logits = torch.randn(embeddings.size(0), 10)  # Placeholder for actual classifier
                    predictions = torch.argmax(logits, dim=1)
                
                labels = batch['label']
            else:
                # Handle other batch formats
                images, labels = batch
                images = images.to(device)
                embeddings = model(images)
                
                # Get predictions (simple classification head)
                logits = torch.randn(embeddings.size(0), 10)  # Placeholder for actual classifier
                predictions = torch.argmax(logits, dim=1)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
            all_predictions.append(predictions.cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    
    return all_embeddings.numpy(), all_labels.numpy(), all_predictions.numpy()