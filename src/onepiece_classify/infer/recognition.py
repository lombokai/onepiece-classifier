from PIL import Image
import numpy as np
import torch
from pathlib import Path
from .base import BaseInference
from typing import Optional, Tuple, Dict

from onepiece_classify.models import ImageRecogModel
from onepiece_classify.data import OnepieceImageDataLoader
from onepiece_classify.transforms import get_test_transforms


class ImageRecognition(BaseInference):
    
    def __init__(self, model_path: str, data_path: str):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)

        self.class_dict = {
            i: val for val, i in OnepieceImageDataLoader(self.data_path).trainset.class_to_idx.items()
        }
        self.name_class = OnepieceImageDataLoader(self.data_path).trainset.classes
        self.nclass = len(self.name_class)
        # self.model = self._build_model()
        
    def _build_model(self):
         # load model
        state_dict = torch.load(self.model_path)
        model_backbone = ImageRecogModel(self.nclass).build_backbone()
        model_backbone.load_state_dict(state_dict)
        return model_backbone
    
    def pre_process(self, image: Optional[str | np.ndarray | Image.Image]) -> torch.Tensor:
        
        trans = get_test_transforms()

        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
            img = trans(img).unsqueeze(0)
        
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
            img = trans(img).unsqueeze(0)

        elif isinstance(image, np.ndarray):
            img = image.astype(np.uint8)
            img = Image.fromarray(img).convert("RGB")
            img = trans(img).unsqueeze(0)

        else:
            print("Image type not recognized")

        return img

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        model = self._build_model()
        model.eval()

        result = model(image_tensor)
        return result
    
    def post_process(self, output: torch.Tensor) -> Tuple[str, float]:
        
        logits_prob = torch.softmax(output, dim=1).squeeze()
        class_idx = int(torch.argmax(logits_prob))
        
        class_names = self.class_dict[class_idx]
        confidence = logits_prob[class_idx]
        return (class_names, float(confidence))
    
    def predict(self, image: Optional[str|np.ndarray|Image.Image]) -> Dict[str, str]:
        
        tensor_img = self.pre_process(image=image)
        logits = self.forward(tensor_img)
        class_names, confidence = self.post_process(logits)

        return {
            "class_names": class_names,
            "confidence": f"{confidence:.4f}"
        }
