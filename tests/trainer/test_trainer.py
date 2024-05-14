import torch
import torch.nn as nn
from torch.optim import Adam

import unittest
from unittest.mock import MagicMock
from torch.utils.data import DataLoader
from pathlib import Path

from onepiece_classify.trainer import Trainer
from onepiece_classify.models import image_recog
from onepiece_classify.data import OnepieceImageDataLoader


class TestOnepieceImageDataLoader(unittest.TestCase):
    def setUp(self):
        self.root_path = "data"
        self.batch_size = 32
        self.num_workers = 0

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        loader = OnepieceImageDataLoader(
            self.root_path,
            self.batch_size,
            self.num_workers
        )
        self.nclass = len(loader.trainset.classes)
        model = image_recog(num_classes=self.nclass).to(self.device)

        lrate = 0.001
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lrate)
        self.trainer = Trainer(
            max_epochs = 2,
            loss_fn = loss_fn,
            optim = optimizer,
            learning_rate = lrate
        )

        self.trainer.model = model
        self.trainer.loader = loader
        self.trainer.path_to_save = Path("checkpoint/model_checkpoint.pth")

    def test_train_step(self):
        fake_batch_x = torch.rand(1, 3, 224, 224).to(self.device)
        fake_batch_y = torch.randint(0, self.nclass, size=(1,)).to(self.device)
        batch = (fake_batch_x, fake_batch_y)

        _loss, _acc = self.trainer.train_step(batch)

        self.assertTrue(isinstance(_loss, float))
        self.assertTrue(isinstance(_acc, float))
        self.assertTrue(_loss > 0.)

    def test_val_step(self):
        fake_batch_x = torch.rand(1, 3, 224, 224).to(self.device)
        fake_batch_y = torch.randint(0, self.nclass, size=(1,)).to(self.device)
        batch = (fake_batch_x, fake_batch_y)

        _loss, _acc = self.trainer.train_step(batch)

        self.assertTrue(isinstance(_loss, float))
        self.assertTrue(isinstance(_acc, float))
        self.assertTrue(_loss > 0.)

    def test_train_batch(self):
        _loss, _acc = self.trainer.train_batch()

        self.assertTrue(isinstance(_loss, float))
        self.assertTrue(isinstance(_acc, float))
        self.assertTrue(_loss > 0.)
        self.assertTrue((_acc > 0) and (_acc < 1))

    def test_val_batch(self):
        _loss, _acc = self.trainer.val_batch()

        self.assertTrue(isinstance(_loss, float))
        self.assertTrue(isinstance(_acc, float))
        self.assertTrue(_loss > 0.)
        self.assertTrue((_acc > 0) and (_acc < 1))

    def test_init_and_run(self):
        self.trainer.run()

        self.assertTrue(len(self.trainer._train_loss) > 0)
        self.assertTrue(len(self.trainer._train_acc) > 0)
        self.assertTrue(len(self.trainer._val_loss) > 0)
        self.assertTrue(len(self.trainer._val_acc) > 0)

        self.assertTrue(self.trainer.train_loss > 0)
        self.assertTrue(self.trainer.train_acc > 0)
        self.assertTrue(self.trainer.val_loss > 0)
        self.assertTrue(self.trainer.val_acc > 0)

        self.assertTrue(self.trainer.path_to_save.is_file())