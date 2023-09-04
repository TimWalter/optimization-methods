from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
from scipy.sparse.linalg import LinearOperator

from src.regularizer import Regularizer, DfRegularizer, PfRegularizer, L2SRegularizer, \
    L21Regularizer, L1Regularizer
from src.linear_operators import IdentityOperator, StackedOperator
from src.preprocessing import estimate_initial_intensity
from src.proximal_operators import L2SOperator, TranslationOperator, ScaledOperator, \
    SeparableSumOperator, IndicatorOperator


class Formulation(ABC):
    """Abstract base class for problems"""

    @abstractmethod
    def __init__(self, a: LinearOperator, b: np.ndarray, **kwargs) -> None:
        self.a = a
        self.b = b

    @abstractmethod
    def f(self, x: np.ndarray) -> float:
        """Returns function value"""
        pass


class PfFormulation(Formulation, ABC):
    """Abstract base class for proximal-friendly problems """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def partial_df(self, x: np.ndarray) -> np.ndarray:
        """Returns partial derivative"""
        pass

    @abstractmethod
    def proximal_operator(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """Apply proximal operator without linear operator"""
        pass

    @abstractmethod
    def proximal_operator_lo(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """Apply proximal operator with linear operator"""
        pass

    @abstractmethod
    def get_linear_operators_in_pol(self) -> List[LinearOperator]:
        """Returns linear operators used in the proximal operator with linear operator"""
        pass


class DfFormulation(Formulation, ABC):
    """Abstract base class for differentiable problems"""

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def df(self, x: np.ndarray) -> np.ndarray:
        """Returns derivative"""
        pass


class LSFormulation(DfFormulation):
    """Least squares problem """

    def __init__(self, a: LinearOperator, b: np.ndarray) -> None:
        """
        Args:
            a (np.ndarray): Linear operator
            b (np.ndarray): Right-hand side vector
        """
        super().__init__(a, b)
        self.proximal_operator_lo = ScaledOperator(0.5, TranslationOperator(b, L2SOperator()))

    def f(self, x: np.ndarray) -> float:
        return 0.5 * np.linalg.norm(self.a @ x - self.b) ** 2

    def df(self, x: np.ndarray) -> np.ndarray:
        return self.a.T @ (self.a @ x - self.b)

    def proximal_operator_lo(self, x: np.ndarray, sigma: float) -> np.ndarray:
        return self.proximal_operator_lo(x, sigma)


class BaseLSRFormulation(Formulation):
    """Least squares problem with regularization base class"""

    def __init__(self, a: LinearOperator, b: np.ndarray, beta: List[float] = None,
                 regularizer: List[Regularizer] = None,
                 lo: List[LinearOperator] = None) -> None:
        """
        Args:
            a (np.ndarray): Linear operator
            b (np.ndarray): Right-hand side vector
            beta (List[float], optional): Regularization strength. Defaults to 1.
            regularizer (List[Regularizer], optional): Regularizer. Defaults to L2SRegularizer().
            lo (List[LinearOperator], optional): Linear operator on the solution in the norm. Defaults to identity.
        """
        super().__init__(a, b)

        self.least_squares = LSFormulation(a, b)

        self.beta = beta if beta else [1]
        self.regularizer = regularizer if regularizer else [L2SRegularizer()] * len(self.beta)
        self.lo = lo if lo else [IdentityOperator((self.a.shape[1], self.a.shape[1]))] * len(self.beta)

    def f(self, x: np.ndarray) -> float:
        return self.least_squares.f(x) + np.sum([beta * regularizer(lo @ x) for beta, regularizer, lo in
                                                 zip(self.beta, self.regularizer, self.lo)], axis=0)


class LSDfRFormulation(BaseLSRFormulation, DfFormulation):
    """Least squares problem with a differentiable regularization"""

    def __init__(self, a: LinearOperator, b: np.ndarray, beta: List[float] = None,
                 regularizer: List[DfRegularizer] = None,
                 lo: List[LinearOperator] = None) -> None:
        super().__init__(a, b, beta, regularizer, lo)

    def df(self, x: np.ndarray) -> np.ndarray:
        return self.least_squares.df(x) + np.sum([beta * lo.T @ regularizer.df(lo @ x) for beta, regularizer, lo in
                                                  zip(self.beta, self.regularizer, self.lo)], axis=0)


class LSPfRFormulation(BaseLSRFormulation, PfFormulation):
    """Least squares problem with a proximal friendly regularization"""

    def __init__(self, a: LinearOperator, b: np.ndarray, beta: List[float] = None,
                 regularizer: List[Regularizer] = None,
                 lo: List[LinearOperator] = None) -> None:
        """
        Notes:
            Only the last regularizer can be proximal friendly the rest should be differentiable
        """
        super().__init__(a, b, beta, regularizer, lo)

        proximal_operators = []
        proximal_operators_lo = [ScaledOperator(0.5, TranslationOperator(self.b, L2SOperator()))]
        proximal_operators_lo_slices = [self.a.shape[0]]
        self.linear_operators_in_pol = [self.a]
        for beta, regularizer, lo in zip(self.beta, self.regularizer, self.lo):
            if isinstance(regularizer, PfRegularizer):
                if isinstance(lo, IdentityOperator):
                    proximal_operators.append(ScaledOperator(beta, regularizer.get_proximal_operator()))
                else:
                    proximal_operators_lo.append(ScaledOperator(beta, regularizer.get_proximal_operator()))
                    proximal_operators_lo_slices.append(lo.shape[0])
                    self.linear_operators_in_pol.append(lo)

        if not proximal_operators:
            proximal_operators = [IndicatorOperator(0, 1)]
        self.proximal_operator = SeparableSumOperator(proximal_operators, [self.a.shape[1]] * len(proximal_operators))
        self.proximal_operator_lo = SeparableSumOperator(proximal_operators_lo, proximal_operators_lo_slices)

    def partial_df(self, x: np.ndarray) -> np.ndarray:
        return self.least_squares.df(x) + np.sum([beta * lo.T @ regularizer.df(lo @ x) for beta, regularizer, lo in
                                                  zip(self.beta[:-1], self.regularizer[:-1], self.lo[:-1])], axis=0)

    def proximal_operator(self, x: np.ndarray, sigma: float) -> np.ndarray:
        return self.proximal_operator(x, sigma)

    def proximal_operator_lo(self, x: np.ndarray, sigma: float) -> np.ndarray:
        return self.proximal_operator_lo(x, sigma)

    def get_linear_operators_in_pol(self) -> List[LinearOperator]:
        return self.linear_operators_in_pol


class TikhonovFormulation(LSDfRFormulation):
    """Tikhonov problem"""

    def __init__(self, a: LinearOperator, b: np.ndarray, beta: float = 1) -> None:
        """
        Args:
            a (np.ndarray): Linear operator
            b (np.ndarray): Right-hand side vector
            beta (float, optional): Regularization parameter. Defaults to 1.
        """
        super().__init__(a, b, [beta / 2], [L2SRegularizer()])


class LassoFormulation(LSPfRFormulation):
    """Lasso problem"""

    def __init__(self, a: LinearOperator, b: np.ndarray, beta: float = 1) -> None:
        """
        Args:
            a (np.ndarray): Linear operator
            b (np.ndarray): Right-hand side vector
            beta (float, optional): Regularization parameter. Defaults to 1.
        """
        super().__init__(a, b, [beta], [L1Regularizer()])


class ElasticNetFormulation(LSPfRFormulation):
    def __init__(self, a: LinearOperator, b: np.ndarray, beta: List[float] = None) -> None:
        """
        Args:
            a (np.ndarray): Linear operator
            b (np.ndarray): Right-hand side vector
            beta (List[float], optional): Regularization parameters. Defaults to [1, 1].
        """
        beta = beta if beta else [1 / 2, 1]
        super().__init__(a, b, beta, [L2SRegularizer(), L1Regularizer()])


class TVRegularizationFormulation(LSPfRFormulation):
    """Total variation regularization problem"""

    def __init__(self, a: LinearOperator, b: np.ndarray, beta: float, finite_difference_operator: LinearOperator,
                 isotropic: bool = False, shape: tuple = None) -> None:
        """
        Args:
            a (np.ndarray): Linear operator
            b (np.ndarray): Right-hand side vector
            beta (float, optional): Regularization parameter
            finite_difference_operator (np.ndarray): Finite difference operator
            isotropic (bool, optional): Isotropic regularization. Defaults to False
            shape (tuple, optional): Shape of the reconstruction. It Has to be given for isotropic regularization.
        """
        super().__init__(a, b, [beta], [L21Regularizer(shape)] if isotropic else [L1Regularizer()],
                         [finite_difference_operator])

        if isotropic:
            self.sub_df = None

    def sub_df(self, x: np.ndarray) -> np.ndarray:
        return self.partial_df(x) + self.beta * self.lo[0].T @ self.regularizer[0].sub_df(self.lo[0] @ x)


class TransmissionLogLikelihoodFormulation(DfFormulation):
    """Transmission log likelihood problem"""

    def __init__(self, a: LinearOperator, b: np.ndarray, blank_scans: np.ndarray = None,
                 n_background_events: np.ndarray = None) -> None:
        """
        Args:
            a (np.ndarray): X-ray transform
            b (np.ndarray): Sinogram
            blank_scans (np.ndarray): Blank scans. Defaults to the initial intensity.
            n_background_events (int, optional): Number of background events. Default to 0.
        """
        self.a = a
        self.b = b
        if blank_scans is None:
            blank_scans = estimate_initial_intensity(b)
        self.blank_scans = blank_scans
        if n_background_events is None:
            n_background_events = np.zeros_like(b)
        self.n_background_events = n_background_events

    def f(self, x: np.ndarray) -> float:
        likelihood = self.blank_scans * np.exp(- self.a @ x) + self.n_background_events
        return float(np.sum(likelihood - self.b * np.log(likelihood)))

    def df(self, x: np.ndarray) -> np.ndarray:
        likelihood = self.blank_scans * np.exp(-self.a @ x)
        return self.a.T @ (likelihood * (self.b / (likelihood + self.n_background_events)) - 1)
