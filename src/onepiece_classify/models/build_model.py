import torch 
import torch.nn as nn
from torchvision import models


# def create_model(
#     num_classes: int,
#     seed: int=42
# ):
#     model = models.mobilenet_v3_large(weights="DEFAULT")

#     for param in model.parameters():
#         param.required_grad = False

#     # change model head
#     torch.manual_seed(seed)
#     model.classifier = nn.Sequential(
#         nn.Dropout(p=0.2),
#         nn.Linear(in_features=960, out_features=num_classes)
#     )

#     return model

class ImageRecogModel(nn.Module):
    
    def __init__(self, in_features, num_classes):
        self.in_features = in_features
        self.num_classes = num_classes

        self.backbone = _build_backbone()
        self.dropout = nn.Dropout(p=0.2)

    def _build_backbone(self) -> models:
        model = models.mobilenet_v3_large(weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False

        return model

    def forward(self) -> torch.Tensor:
        # x = self.backbone()
        # x = self.dropout(nn.Linear(self.in_features, self.num_classes))
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.in_features, self.num_classes)
        )

        return self.backbone.classifier

def create_model(in_features):
    net = ImageRecogModel(960, 18)
