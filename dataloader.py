import os
import numpy as np
import torch
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

train_batch_size = 32
test_batch_size = 32
train_size_rate = 0.8   

data_transforms = transforms.Compose([
    transforms.RandomRotation(90),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def make_train_dataloader(data_path):
    dataset = datasets.ImageFolder(root=data_path) 
    train_size = int(len(dataset) * train_size_rate)
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_dataset.dataset.transform = data_transforms
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
    valid_dataset.dataset.transform = test_transforms
    valid_loader = DataLoader(valid_dataset, batch_size=train_batch_size, shuffle=False, num_workers=0)
    return train_loader, valid_loader

class TestDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.image_files = os.listdir(data_path)  
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.image_files[idx])
        img = Image.open(img_path).convert('RGB') 
        if self.transform:
            img = self.transform(img)  
        return img

def make_test_dataloader(data_path, batch_size=32, num_workers=0):
    test_dataset = TestDataset(data_path, transform=test_transforms)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return test_loader
