import torch

from onepiece_classify.models import image_recog


def test_model():
    rand_tensor = torch.rand([1, 3, 224, 224])
    nclasses = 18

    model = image_recog(num_classes=nclasses)

    out = model.forward(rand_tensor)
    assert out.shape == (1, 18)
