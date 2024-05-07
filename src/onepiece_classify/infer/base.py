from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from PIL import Image
import torch


class BaseInference(ABC):
    
    @abstractmethod
    def pre_process(self, image: Optional[str, np.ndarray, Image])->torch.Tensor:
        pass
    
    @abstractmethod
    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def post_process(self, output: torch.Tensor) -> str:
        pass
    
    @abstractmethod
    def predict(self, image: Optional[str, np.ndarray, Image]) -> dict:
        pass