import unittest
import numpy as np

from src.phantoms import shepp_logan
from src.backprojection import filtered_backprojection
from src.filters import RamLak


class BackprojectionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.sinogram, self.operator, self.ground_truth = shepp_logan((16, 16), 360, 720)

    def test_filtered_backprojection(self):
        projection = filtered_backprojection(
            self.ground_truth.shape,
            self.sinogram,
            np.linspace(0, 360, 720),
            RamLak(),
            self.operator.s2c,
            self.operator.c2d)
        self.assertAlmostEqual(np.mean(self.ground_truth - projection), 0, 1)

    def test_cosine_weighting(self):
        projection = filtered_backprojection(
            self.ground_truth.shape,
            self.sinogram,
            np.linspace(0, 360, 720),
            RamLak(),
            self.operator.s2c,
            self.operator.c2d,
            False)
        self.assertAlmostEqual(np.mean(self.ground_truth - projection), 0, 1)


if __name__ == '__main__':
    unittest.main()
