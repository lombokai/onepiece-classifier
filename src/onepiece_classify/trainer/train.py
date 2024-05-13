import torch
import torch.nn as nn
from torchvision import transforms
from src.onepiece_classify.data.data_setup import create_dataloaders
from src.onepiece_classify.models.build_model import create_model
from src.onepiece_classify.trainer.engine import train


device = "cuda" if torch.cuda.is_available() else "cpu"

# data transformation
train_trans = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406], 
        std = [0.229, 0.224, 0.225],
    )
])

val_trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

# params
batch_size = 32
train_dir = "src/data/train/"
val_dir = "src/data/val/"

# create data loaders
train_loader, val_loader, class_names = create_dataloaders(
    train_dir=train_dir,
    val_dir=val_dir,
    train_transform=train_trans,
    val_transform=val_trans,
    batch_size=batch_size
)

# put model to device
model = create_model(num_classes=len(class_names)).to(device)

# hyperparameters
epochs = 5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def run_training():
    train(
        model=model,
        loss_fn=criterion,
        optim=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=5
    )

if __name__ == "__main__":
    run_training()
