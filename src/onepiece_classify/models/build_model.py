import torch 
import torch.nn as nn
from torchvision import models


def create_model(
    num_classes: int,
    seed: int=42
):
    model = models.mobilenet_v3_large(weights="DEFAULT")

    for param in model.parameters():
        param.required_grad = False

    # change model head
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=960, out_features=num_classes)
    )

    return model