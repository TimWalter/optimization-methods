import numpy as np
from abc import ABC, abstractmethod

from src.proximal_operators import L2SOperator, L21Operator, L1Operator, HuberOperator, ProximalOperator


class Regularizer(ABC):
    """Abstract base class for regularizers"""

    @abstractmethod
    def __call__(self, x):
        """Returns function value"""
        pass


class DfRegularizer(Regularizer, ABC):
    """Abstract base class for differentiable regularizers"""

    @abstractmethod
    def df(self, x):
        """Returns derivative"""
        pass


class PfRegularizer(Regularizer, ABC):
    """Abstract base class for proximal-friendly regularizers"""

    @abstractmethod
    def get_proximal_operator(self) -> ProximalOperator:
        """Returns proximal operator"""
        pass


class L2SRegularizer(DfRegularizer, PfRegularizer):
    """Regularizer with the squared L2 norm """

    def __init__(self):
        self.proximal_operator = L2SOperator()

    def __call__(self, x):
        return np.linalg.norm(x) ** 2

    def df(self, x):
        return 2 * x

    def get_proximal_operator(self) -> ProximalOperator:
        return self.proximal_operator


class L21Regularizer(PfRegularizer):
    """Regularizer with the L21 norm"""

    def __init__(self, shape):
        self.proximal_operator = L21Operator()
        self.shape = shape

    def __call__(self, x):
        return np.linalg.norm(x.reshape(self.shape), ord=2, axis=1).sum()

    def get_proximal_operator(self) -> ProximalOperator:
        return self.proximal_operator


class L1Regularizer(PfRegularizer):
    """Regularizer with the L1 norm"""

    def __init__(self):
        self.proximal_operator = L1Operator()

    def __call__(self, x):
        return np.linalg.norm(x, ord=1)

    def get_proximal_operator(self) -> ProximalOperator:
        return self.proximal_operator

    def sub_df(self, x):
        return np.sign(x)


class HuberRegularizer(DfRegularizer, PfRegularizer):
    """Regularizer with the Huber Loss"""

    def __init__(self, delta):
        if delta <= 0:
            raise ValueError("Delta must be positive")
        self.delta = delta
        self.proximal_operator = HuberOperator(delta)

    def __call__(self, x):
        return np.sum(np.where(
            np.abs(x) < self.delta,
            0.5 * x ** 2,
            self.delta * (np.abs(x) - 0.5 * self.delta)
        ))

    def df(self, x):
        return np.where(
            np.abs(x) < self.delta,
            x,
            self.delta * np.sign(x)
        )

    def get_proximal_operator(self) -> ProximalOperator:
        return self.proximal_operator


class FairRegularizer(DfRegularizer):
    """Regularizer with the Fair Potential"""

    def __init__(self, delta):
        self.delta = delta

    def __call__(self, x):
        return np.sum(self.delta ** 2 * (np.abs(x / self.delta) - np.log(1 + np.abs(x / self.delta))))

    def df(self, x):
        return x / (1 + np.abs(x / self.delta))
