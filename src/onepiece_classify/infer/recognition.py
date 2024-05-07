



from .base import BaseInference
import torch

class ImageRecognition(BaseInference):
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        
        self.model = self._build_model(model_path)
        
    def _build_model(self, model_path):
         # load model
        state_dict = torch.load(model_path)

        model = ImageModel.load(state_dict)
        return model
    
    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self.model.forward(image_tensor)
    
    
    
def recognition():
    return ImageRecognition(model_path="src/checkpoint/checkpoint_notebook.pth")