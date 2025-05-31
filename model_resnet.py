import torch
from torchvision import models
import torch.nn as nn

def build_model(num_classes):
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), device
