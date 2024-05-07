from ..src.onepiece_classify.models import build_model

def test_model():

    assert build_model.model() == torch.nn.Module