import unittest
import numpy as np

from src.linear_operators import StackedOperator, IdentityOperator


class StackedOperatorTestCase(unittest.TestCase):
    def test_stacked_operator(self):
        operator = StackedOperator([np.random.rand(3, 4), np.random.rand(2, 4)])
        test = np.arange(4)
        result = operator @ test
        self.assertTrue(result.shape == (5,))


if __name__ == '__main__':
    unittest.main()
