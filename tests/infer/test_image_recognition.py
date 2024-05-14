import torch
import unittest
import numpy as np
from PIL import Image

from onepiece_classify.infer import ImageRecognition


class TestImageRecognition(unittest.TestCase):
    def setUp(self) -> None:
        
        self.model_path = "checkpoint/checkpoint.pth"
        self.data_path = "data"
        self.image_to_test = "data/test/Ace/212.png_inverted.png"

        self.recog = ImageRecognition(
            model_path = self.model_path,
            data_path = self.data_path
        )

        self.num_class = self.recog.nclass
        self.name_class = self.recog.name_class


    def test_preprocess(self):
        img_path = self.image_to_test
        pil_image = Image.open(self.image_to_test)
        fake_numpy_image = np.random.random((326, 278, 3))

        res_img_path = self.recog.pre_process(img_path)
        res_pil_image = self.recog.pre_process(pil_image)
        res_numpy_image = self.recog.pre_process(fake_numpy_image)

        self.assertTrue(isinstance(res_img_path, torch.Tensor))
        self.assertTrue(isinstance(res_pil_image, torch.Tensor))
        self.assertTrue(isinstance(res_numpy_image, torch.Tensor))

        self.assertEqual(res_img_path.shape, (1, 3, 224, 224))
        self.assertEqual(res_pil_image.shape, (1, 3, 224, 224))
        self.assertEqual(res_numpy_image.shape, (1, 3, 224, 224))

    def test_forward(self):
        fake_image = torch.randn(1, 3, 224, 224)
        result = self.recog.forward(fake_image)

        self.assertEqual(result.shape, (1, self.num_class))

    def test_post_process(self):
        fake_logits = torch.randn(1, self.num_class)
        result_1, result_2 = self.recog.post_process(fake_logits)

        self.assertTrue(isinstance(result_1, str))
        self.assertTrue(isinstance(result_2, float))
        self.assertTrue(result_1 in self.name_class)
        self.assertTrue(result_2 >= 0 and result_2 <= 1)

    def test_predict(self):
        result = self.recog.predict(self.image_to_test)

        self.assertTrue(isinstance(result, dict))
