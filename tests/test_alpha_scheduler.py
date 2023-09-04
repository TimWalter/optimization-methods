import unittest
import numpy as np
from src.optimizer import Optimizer
from src.alpha_scheduler import Decay, Backtracking, Wolfe, BarzilaiBorwein


class AlphaSchedulerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        def f(x):
            return np.sqrt(x ** 2 + 5)

        def df(x):
            return x / np.sqrt(x ** 2 + 5)

        self.f = f
        self.df = df

        x0 = np.array(1)
        self.solution = 0
        self.optimizer = Optimizer(f, df, x0, 4, patience=10, max_iter=10000, tol=1e-8)

    def test_constant(self):
        print("---Constant---")
        result = self.optimizer.gradient_descent()
        self.assertAlmostEqual(np.abs(result - self.solution), 0, 2)

    def test_decay(self):
        print("---Decay---")
        self.optimizer.alpha_scheduler = Decay(0.95)
        result = self.optimizer.gradient_descent()
        self.assertAlmostEqual(np.abs(result - self.solution), 0, 2)

    def test_backtracking(self):
        print("---Backtracking---")
        self.optimizer.alpha_scheduler = Backtracking(self.f, self.df)
        result = self.optimizer.gradient_descent()
        self.assertAlmostEqual(np.abs(result - self.solution), 0, 2)

    def test_wolfe(self):
        print("---Wolfe---")
        self.optimizer.alpha_scheduler = Wolfe(self.f, self.df)
        result = self.optimizer.gradient_descent()
        self.assertAlmostEqual(np.abs(result - self.solution), 0, 2)

    def test_barzilai_borwein(self):
        print("---Barzilai-Borwein---")
        self.optimizer.alpha_scheduler = BarzilaiBorwein(self.f, self.df, "short")
        result = self.optimizer.gradient_descent()
        self.assertAlmostEqual(np.abs(result - self.solution), 0, 2)
        self.optimizer.alpha_scheduler = BarzilaiBorwein(self.f, self.df, "long")
        result = self.optimizer.gradient_descent()
        self.assertAlmostEqual(np.abs(result - self.solution), 0, 2)


if __name__ == '__main__':
    unittest.main()
