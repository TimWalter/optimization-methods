from abc import ABC, abstractmethod
from typing import Callable
from scipy.optimize import line_search
import numpy as np


class AlphaScheduler(ABC):
    """Abstract base class for alpha scheduler"""

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, a_prev: float, x_new: np.ndarray, x: np.ndarray, i: int) -> float:
        """Returns alpha for given iteration"""
        pass


class Constant(AlphaScheduler):
    """Constant alpha scheduler"""

    def __init__(self, **kwargs) -> None:
        pass

    def __call__(self, a_prev: float, _: np.ndarray, __: np.ndarray, ___: int) -> float:
        return a_prev


class Decay(AlphaScheduler):
    """Linear decay Alpha scheduler"""

    def __init__(self, decay_rate: float) -> None:
        """
        Args:
            decay_rate (float): Decay rate
        """
        self.decay_rate = decay_rate

    def __call__(self, a_prev: float, _: np.ndarray, __: np.ndarray, ___: int) -> float:
        return a_prev * self.decay_rate


class Custom(AlphaScheduler):
    """Custom alpha scheduler"""

    def __init__(self, f: Callable) -> None:
        """
        Args:
            f (Callable): Scheduling function
        """
        self.f = f

    def __call__(self, a_prev: float, x_new: np.ndarray, x: np.ndarray, i: int) -> float:
        return self.f(a_prev, x_new, x, i)


class Backtracking(AlphaScheduler):
    """Backtracking alpha scheduler"""

    def __init__(self, f: Callable, df: Callable, rho: float = 0.5, c: float = 0.5, min_alpha: float = 1e-8) -> None:
        """
        Args:
            f (Callable): Function to minimize
            df (Callable): Derivative of function to minimize
            rho (float, optional): Factor to decrease alpha by. Defaults to 0.5.
            c (float, optional): Factor of desired decrease in f. Defaults to 0.5.
            min_alpha (float, optional): Minimum alpha. Defaults to 1e-8.
        """
        self.f = f
        self.df = df
        self.rho = rho
        self.c = c
        self.min_alpha = min_alpha

    def __call__(self, a_prev: float, x_new: np.ndarray, _: np.ndarray, __: int) -> float:
        g = self.df(x_new)
        g_norm = np.linalg.norm(g) ** 2
        loss_prev = self.f(x_new)
        alpha = a_prev
        while self.f(x_new - alpha * g) > loss_prev - self.c * alpha * g_norm:
            alpha *= self.rho
            if alpha < self.min_alpha:
                print("Alpha too small")
                break
        return alpha


class Wolfe(AlphaScheduler):
    """Wolfe alpha scheduler"""

    def __init__(self, f: Callable, df: Callable, c1: float = 1e-4, c2: float = 0.9,
                 min_alpha: float = 1e-8) -> None:
        """
        Args:
            f (Callable): Function to minimize
            df (Callable): Derivative of function to minimize
            c1 (float, optional): Parameter for Armijo condition rule.  Defaults to 1e-4.
            c2 (float, optional): Parameter for curvature condition rule. Defaults to 0.9.
            min_alpha (float, optional): Minimum alpha. Defaults to 1e-8.
        """
        self.f = f
        self.df = df
        self.c1 = c1
        self.c2 = c2
        self.min_alpha = min_alpha

    def __call__(self, a_prev: float, x_new: np.ndarray, _: np.ndarray, __: int) -> float:
        alpha = line_search(self.f, self.df, x_new, -self.df(x_new), c1=self.c1, c2=self.c2)[0]
        if alpha is None:
            print("Line search failed, using previous alpha")
            alpha = a_prev
        return alpha


class BarzilaiBorwein(AlphaScheduler):
    """Barzilai-Borwein alpha scheduler"""

    def __init__(self, f: Callable, df: Callable, option: str, min_alpha: float = 1e-8) -> None:
        """
        Args:
            f (Callable): Function to minimize
            df (Callable): Derivative of function to minimize
            option (str): Either "short" or "long". Whether to use the long or short step
            min_alpha (float, optional): Minimum alpha. Defaults to 1e-8.
        """
        self.f = f
        self.df = df
        self.option = option
        self.min_alpha = min_alpha

    def __call__(self, a_prev: float, x_new: np.ndarray, x: np.ndarray, _: int) -> float:
        g = self.df(x_new)
        g_prev = self.df(x)
        x_diff = x_new - x
        g_diff = g - g_prev
        if self.option == "short":
            alpha = np.dot(x_diff.T, g_diff) / (np.dot(g_diff.T, g_diff) + np.finfo(float).eps)
        elif self.option == "long":
            alpha = np.dot(x_diff.T, x_diff) / (np.dot(x_diff.T, g_diff) + np.finfo(float).eps)
        else:
            raise ValueError("option must be either 'short' or 'long'")
        if alpha < self.min_alpha:
            print(f"Barzilai-Borwein alpha too small {alpha}, using previous alpha")
            alpha = a_prev

        return alpha
