import torch 
import torch.nn as nn
from torchvision import models


class ImageRecogModel(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.backbone = self._build_backbone()
        self.in_features = self._build_backbone().classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.in_features, out_features=self.num_classes)
        )

        # self.dropout = nn.Dropout(0.2)

    def _build_backbone(self):
        model = models.mobilenet_v3_large(weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False
        
        return model

    def forward(self, x):
        x = self.backbone(x)        
        return x

def image_recog(num_classes):
    net = ImageRecogModel(num_classes)
    return net
