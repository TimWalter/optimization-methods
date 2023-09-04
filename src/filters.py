import numpy as np
from abc import ABC, abstractmethod

"""Filtering functions for the backprojection algorithm"""


class Filter(ABC):
    @abstractmethod
    def __call__(self, size):
        pass


class RamLak(Filter):
    def __call__(self, size):
        unit_f = np.linspace(-1, 1, size)
        return np.abs(unit_f)


class SheppLogan(Filter):
    def __call__(self, size):
        unit_f = np.linspace(-1, 1, size)
        return np.abs(unit_f) * np.sinc(unit_f / 2)


class Cosine(Filter):
    def __call__(self, size):
        unit_f = np.linspace(-1, 1, size)
        return np.abs(unit_f) * np.cos(unit_f * np.pi / 2)


class Gaussian(Filter):
    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def __call__(self, size):
        unit_f = np.linspace(-1, 1, size)
        return np.abs(unit_f) * np.exp(-np.pi * unit_f ** 2 / (2 * self.sigma ** 2))
