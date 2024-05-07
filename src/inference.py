import torch
from predict import predict


model_path = "src/checkpoint/checkpoint_notebook.pth"
image_path = "src/data/test/Akainu/modified_203.png"
device = "cuda" if torch.cuda.is_available() else "cpu"

name = predict(model_path=model_path, image_path=image_path, device=device)
print(name)