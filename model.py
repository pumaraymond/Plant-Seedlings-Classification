import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
def my_resnet_50():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 12))
    return model
