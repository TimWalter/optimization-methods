import unittest
import numpy as np
from src.optimizer import Optimizer
from src.formulations import symmetric_quadratic, lasso


class OptimizerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        def f(x):
            return np.sqrt(x ** 2 + 5)

        def df(x):
            return x / np.sqrt(x ** 2 + 5)

        x0 = np.array(1)
        self.solution = 0
        self.optimizer = Optimizer(f, df, x0, 1e-2, patience=10, max_iter=10000)

    def test_verbose(self):
        print("---Verbose---")
        self.optimizer.verbose = True
        result = self.optimizer.adam()
        self.assertEqual(len(result), 3, "Result should be a tuple of length 3")
        self.optimizer.verbose = False
        result = self.optimizer.adam()
        self.assertIsInstance(result, np.ndarray | float, "Result should be a numpy array")

    def test_gradient_descent(self):
        print("---Gradient Descent---")
        result = self.optimizer.gradient_descent()
        self.assertAlmostEqual(np.abs(result - self.solution), 0, 2)

    def test_momentum(self):
        print("---Momentum---")
        result = self.optimizer.momentum()
        self.assertAlmostEqual(np.abs(result - self.solution), 0, 2)

    def test_adam(self):
        print("---Adam---")
        result = self.optimizer.adam()
        self.assertAlmostEqual(np.abs(result - self.solution), 0, 3)

    def test_optimized_gradient(self):
        print("---Optimized Gradient---")
        result = self.optimizer.optimized_gradient()
        self.assertAlmostEqual(np.abs(result - self.solution), 0, 1)

    def test_nesterov_momentum(self):
        print("---Nesterov Momentum---")
        result = self.optimizer.nesterov_momentum()
        self.assertAlmostEqual(np.abs(result - self.solution), 0, 2)

    def test_fast_gradient(self):
        print("---Fast Gradient---")
        result = self.optimizer.fast_gradient()
        self.assertAlmostEqual(np.abs(result - self.solution), 0, 1)

    def test_conjugate_gradient(self):
        print("---Conjugate Gradient---")
        a = np.eye(100)
        b = np.arange(100)
        f, df = symmetric_quadratic(a, b)
        x0 = np.random.rand(100)
        optimizer = Optimizer(f, df, x0, 1e-2, patience=10, max_iter=1000)
        result = optimizer.conjugate_gradient(a.T @ a)
        self.assertAlmostEqual(np.linalg.norm(a @ result - b), 0, 2)

    def test_ista(self):
        print("---ISTA---")
        a = np.eye(100)
        b = np.zeros(100)
        b[0] = 1
        f, df = lasso(a, b)
        x0 = np.random.rand(100)
        optimizer = Optimizer(f, df, x0, 1e-2, patience=10, max_iter=1000)
        result = optimizer.ista()
        self.assertLessEqual(f(result), f(b))

    def test_projected_gradient_descent(self):
        print("---Projected Gradient Descent---")
        a = np.eye(100)
        b = np.zeros(100)
        b[0] = 1
        f, df = lasso(a, b)
        x0 = np.random.rand(100)
        optimizer = Optimizer(f, df, x0, 1e-2, patience=10, max_iter=1000)

        def projection(x):
            return np.maximum(x, 0)

        result = optimizer.projected_gradient_descent(projection)
        self.assertLessEqual(f(result), f(x0))


if __name__ == '__main__':
    unittest.main()
