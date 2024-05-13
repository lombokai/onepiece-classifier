from PIL import Image
from numpy import ndarray
import torch
from pathlib import Path
from .base import BaseInference

from onepiece_classify.models import ImageRecogModel
from onepiece_classify.data import OnepieceImageDataLoader
from onepiece_classify.transforms import get_test_transforms


class ImageRecognition(BaseInference):
    
    def __init__(self, model_path: str, data_path: str):
        self.model_path = Path(model_path)
        self.data_path = data_path

        self.nclass = len(OnepieceImageDataLoader(self.data_path).trainset.classes)
        self.model = self._build_model(str(model_path))
        
    def _build_model(self, model_path):
         # load model
        state_dict = torch.load(model_path)
        model_backbone = ImageRecogModel(self.nclass).backbone
        model = model_backbone.load(state_dict)
        return model
    
    def pre_process(self, image: Optional[str, ndarray, Image]) -> torch.Tensor:
        if type(image) == "str":
            img = Image.open(image).convert("RGB")

            trans = get_test_transforms()

            img = trans(img).unsqueeze(0)

            model = self.model.eval()

            return torch.argmax(model(img))
        
        elif type(image) == Image:

            trans = get_test_transforms()

            img = trans(img).unsqueeze(0)

            model = self.model.eval()

            return torch.argmax(model(img))
        
        elif type(image) == ndarray:

            trans = get_test_transforms()

            img = trans(img).unsqueeze(0)

            model = self.model.eval()

            return torch.argmax(model(img))

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        result = self.model.forward(image_tensor)
        return result
    
    def post_process(self, output: torch.Tensor) -> str:
        pass
    
    def predict(self, image: Optional[str, np.ndarray, Image]) -> dict:
        return super().predict(image)
    
    
    
def recognition():
    return ImageRecognition(model_path="src/checkpoint/checkpoint_notebook.pth")

if __name__ == "__main__":
    recognition()