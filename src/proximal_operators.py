from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np


class ProximalOperator(ABC):
    """Abstract base class for proximal operators"""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """Returns proximal operator for given iteration"""
        pass


class IndicatorOperator(ProximalOperator):
    """Indicator proximal operator"""

    def __init__(self, min_val: float = 0, max_val: float = 1) -> None:
        """
        Args:
            min_val (float): Minimum value
            max_val (float): Maximum value
        """
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x: np.ndarray, _: float) -> np.ndarray:
        return np.clip(x, self.min_val, self.max_val)


class ConstantFunctionalOperator(ProximalOperator):
    """Constant functional proximal operator"""

    def __init__(self) -> None:
        pass

    def __call__(self, x: np.ndarray, _: float) -> np.ndarray:
        return x


class TranslationOperator(ProximalOperator):
    """Translation proximal operator"""

    def __init__(self, translation: np.ndarray, non_translated_proximal_operator: ProximalOperator) -> None:
        """
        Args:
            translation (np.ndarray): Translation
            non_translated_proximal_operator (ProximalOperator): Non translated proximal operator
        """
        self.translation = translation
        self.non_translated_proximal_operator = non_translated_proximal_operator

    def __call__(self, x: np.ndarray, sigma: float) -> np.ndarray:
        return self.translation + self.non_translated_proximal_operator(x - self.translation, sigma)


class ScaledOperator(ProximalOperator):
    """Scaled proximal operator"""

    def __init__(self, coefficient: float, non_scaled_proximal_operator: ProximalOperator) -> None:
        """
        Args:
            coefficient (float): Coefficient
        """
        self.coefficient = coefficient
        self.non_scaled_proximal_operator = non_scaled_proximal_operator

    def __call__(self, x: np.ndarray, sigma: float) -> np.ndarray:
        return self.non_scaled_proximal_operator(x, sigma * self.coefficient)


class L1Operator(ProximalOperator):
    """L1 norm proximal operator"""

    def __init__(self) -> None:
        pass

    def __call__(self, x: np.ndarray, sigma: float) -> np.ndarray:
        return np.sign(x) * np.maximum(np.abs(x) - sigma, 0)


class L2SOperator(ProximalOperator):
    """L2 norm proximal operator"""

    def __init__(self) -> None:
        pass

    def __call__(self, x: np.ndarray, sigma: float) -> np.ndarray:
        return x / (1 + sigma)


class L21Operator(ProximalOperator):
    """L21 norm proximal operator"""

    def __init__(self) -> None:
        pass

    def __call__(self, x: np.ndarray, sigma: float) -> np.ndarray:
        return np.max((1 - sigma / (np.linalg.norm(x) + np.finfo(float).eps)), 0) * x


class HuberOperator(ProximalOperator):
    """Huber proximal operator"""

    def __init__(self, delta: float) -> None:
        """
        Args:
            delta (float): Delta parameter of the Huber functional
        """
        self.delta = delta

    def __call__(self, x: np.ndarray, sigma: float) -> np.ndarray:
        return (1 - sigma / (np.max([np.linalg.norm(x), sigma]) + self.delta)) * x


class SeparableSumOperator(ProximalOperator):
    """ Separable sum of proximal operators """

    def __init__(self, proximal_operators: List[ProximalOperator], slices: List[int]) -> None:
        """
        Args:
            proximal_operators (list): List of proximal operators
        """
        self.proximal_operators = proximal_operators
        self.slices = slices

    def __call__(self, x: np.ndarray, sigma: float) -> np.ndarray:
        result = np.zeros_like(x)
        start = 0
        for op, sl in zip(self.proximal_operators, self.slices):
            result[start:start + sl] = op(x[start:start + sl], sigma)
            start += sl
        return result
