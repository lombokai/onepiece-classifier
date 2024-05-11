import torch
import torch.nn as nn
from torch.optim import Adam

import unittest
from unittest.mock import MagicMock
from torch.utils.data import DataLoader
from onepiece_classify.data import OnepieceImageDataLoader
from pathlib import Path

from onepiece_classify.trainer import Trainer
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

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = image_recog(num_classes=18).to(self.device)

        lrate = 0.001
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=lrate)
        self.trainer = Trainer(
            max_epochs = 10,
            loss_fn = loss_fn,
            optim = optimizer,
            learning_rate = lrate
        )

        self.trainer.model = self.model
        self.trainer.loader = self.loader

        

    def test_train_step(self):
        fake_batch_x = torch.rand(1, 3, 224, 224).to(self.device)
        fake_batch_y = torch.randint(0, 18, size=(1,)).to(self.device)
        batch = (fake_batch_x, fake_batch_y)

        _loss, _acc = self.trainer.train_step(batch)

        self.assertEqual(type(_loss), float)
        self.assertEqual(type(_acc), float)
        self.assertTrue(_loss > 0.)
        # self.assertEqual((_acc > 0) and (_acc < 1))

    def test_val_step(self):
        fake_batch_x = torch.rand(1, 3, 224, 224).to(self.device)
        fake_batch_y = torch.randint(0, 18, size=(1,)).to(self.device)
        batch = (fake_batch_x, fake_batch_y)

        _loss, _acc = self.trainer.train_step(batch)

        self.assertEqual(type(_loss), float)
        self.assertEqual(type(_acc), float)
        self.assertTrue(_loss > 0.)
        # self.assertTrue((_acc > 0) and (_acc < 1))

    def test_train_batch(self):
        _loss, _acc = self.trainer.train_batch()

        self.assertEqual(type(_loss), float)
        self.assertEqual(type(_acc), float)
        self.assertTrue(_loss > 0.)
        self.assertTrue((_acc > 0) and (_acc < 1))

    def test_val_batch(self):
        _loss, _acc = self.trainer.val_batch()

        self.assertEqual(type(_loss), float)
        self.assertEqual(type(_acc), float)
        self.assertTrue(_loss > 0.)
        self.assertTrue((_acc > 0) and (_acc < 1))

    # def test_init(self):
    #     self.assertTrue(isinstance(trainer.train_loss, list))
    #     self.assertTrue(isinstance(trainer.train_acc, list))
    #     self.assertTrue(isinstance(trainer.val_loss, list))
    #     self.assertTrue(isinstance(trainer.val_acc, list))
        
    # def test_train_step(self):
    #     a, b = train_step(1, 1)
    #     self.assertTrue(isinstance(a, float))
    #     self.assertTrue(isinstance(b, float))
