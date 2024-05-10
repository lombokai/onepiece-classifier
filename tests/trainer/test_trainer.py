import unittest
from unittest.mock import MagicMock
from torch.utils.data import DataLoader
from onepiece_classify.data import OnepieceImageDataLoader
from pathlib import Path

from onepiece_classify.trainer import trainer
from onepiece_classify.models import image_recog

class TestOnepieceImageDataLoader(unittest.TestCase):
    def setUp(self):
        self.root_path = "data"
        self.batch_size = 32
        self.num_workers = 4

        self.loader = OnepieceImageDataLoader(
            self.root_path,
            self.batch_size,
            self.num_workers
        )

        self.model = image_recog(num_classes=18)

        self.training = trainer.fit(self.model, self.loader)

    def test_init(self):
        self.assertTrue(isinstance(trainer.train_loss, list))
        self.assertTrue(isinstance(trainer.train_acc, list))
        self.assertTrue(isinstance(trainer.val_loss, list))
        self.assertTrue(isinstance(trainer.val_acc, list))
    
        
    def test_train_step(self):
        a, b = train_step(1, 1)
        self.assertTrue(isinstance(a, float))
        self.assertTrue(isinstance(b, float))