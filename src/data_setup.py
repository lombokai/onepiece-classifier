from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    batch_size: int
):

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=val_transform)

    # get class names
    class_names = train_data.classes

    # create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader, class_names
