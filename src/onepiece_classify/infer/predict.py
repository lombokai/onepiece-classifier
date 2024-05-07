import torch
import torch.nn as nn
from torchvision import transforms

from PIL import Image
from src.build_model import create_model


class_names = [
    'Ace',
    'Akainu',
    'Brook',
    'Chopper',
    'Crocodile',
    'Franky',
    'Jinbei',
    'Kurohige',
    'Law',
    'Luffy',
    'Mihawk',
    'Nami',
    'Rayleigh',
    'Robin',
    'Sanji',
    'Shanks',
    'Usopp',
    'Zoro'
]

model_base = create_model(num_classes=len(class_names))
test_trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

def predict(
    model_path: str,
    image_path: str,
    device: torch.device
):
    # load model
    state_dict = torch.load(model_path)

    # add model parameters
    model_base.load_state_dict(state_dict)

    # put model to device
    model_base.to(device)

    # load image
    image = Image.open(image_path).convert("RGB")

    # transform and add batch
    img = test_trans(image).unsqueeze(0)

    # eval mode
    model_base.eval()

    # predict
    idx = torch.argmax(model_base(img.to(device)))

    # get class names
    names = class_names[idx]

    return names