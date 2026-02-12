import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import pandas as pd
import numpy as np


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_text_data(data_type='imdb'):
    """Load real text data for drift detection"""
    if data_type == 'imdb':
        # Simulate loading IMDB-like data
        # In practice, you would use datasets.load_dataset('imdb')
        texts = []
        labels = []
        
        # Create sample data (in real implementation, load actual dataset)
        for i in range(1000):
            texts.append(f"Sample text review {i}")
            labels.append(i % 2)  # Binary classification
        
        return texts, labels
    
    raise ValueError(f"Unknown text data type: {data_type}")


def load_image_data(data_type='cifar10', data_dir=None):
    """Load real image data for drift detection"""
    if data_type == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to fit ResNet input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.45], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        return dataset
    
    raise ValueError(f"Unknown image data type: {data_type}")


def get_dataloader(dataset, batch_size=32, shuffle=True):
    """Create dataloader for the dataset"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)