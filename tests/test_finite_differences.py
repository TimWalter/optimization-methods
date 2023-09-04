import unittest
import numpy as np

from src.linear_operators import ForwardDifferences, BackwardDifferences, CentralDifferences


class FiniteDifferencesTestCase(unittest.TestCase):
    def test_forward_differences(self):
        operator = ForwardDifferences((3, 4))
        test = np.arange(12)
        expected = np.array([1, 1, 1, -3, 1, 1, 1, -7, 1, 1, 1, -11, 4, 4, 4, 4, 4, 4, 4, 4, -8, -9, -10, -11])
        res = operator @ test
        self.assertTrue(np.allclose(res, expected))

    def test_backward_differences(self):
        operator = BackwardDifferences((3, 4))
        test = np.arange(12)
        expected = np.array([0, 1, 1, 1, 4, 1, 1, 1, 8, 1, 1, 1, 0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4])
        res = operator @ test
        self.assertTrue(np.allclose(res, expected))

    def test_central_differences(self):
        operator = CentralDifferences((3, 4))
        test = np.arange(12)
        expected = np.array(
            [0.5, 1, 1, -1, 2.5, 1, 1, -3, 4.5, 1, 1, -5, 2, 2.5, 3, 3.5, 4, 4, 4, 4, -2, -2.5, -3, -3.5])
        res = operator @ test
        self.assertTrue(np.allclose(res, expected))


if __name__ == '__main__':
    unittest.main()
