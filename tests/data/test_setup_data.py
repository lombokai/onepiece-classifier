import unittest
from pathlib import Path

from onepiece_classify.data import OnepieceImageDataLoader


class TestOnepieceImageDataLoader(unittest.TestCase):
    def setUp(self):
        self.root_path = "data"
        self.batch_size = 32
        self.num_workers = 4

        self.loader = OnepieceImageDataLoader(
            self.root_path, self.batch_size, self.num_workers
        )

    def test_init(self):
        self.assertEqual(self.loader.root_path, Path(self.root_path))
        self.assertEqual(self.loader.train_path, Path(self.root_path).joinpath("train"))
        self.assertEqual(self.loader.valid_path, Path(self.root_path).joinpath("val"))
        self.assertEqual(self.loader.test_path, Path(self.root_path).joinpath("test"))
        self.assertEqual(self.loader.batch_size, self.batch_size)
        self.assertEqual(self.loader.num_workers, self.num_workers)

    def test_build_dataset_mode_test(self):
        expected_dataset = self.loader._build_dataset(mode="test")
        actual_dataset = self.loader.testset

        self.assertEqual(len(expected_dataset), len(actual_dataset))

        for expected_sample, actual_sample in zip(expected_dataset, actual_dataset):
            self.assertEqual(expected_sample[0].shape, actual_sample[0].shape)
            self.assertEqual(expected_sample[1], actual_sample[1])
