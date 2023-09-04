from typing import Callable, Type
from enum import StrEnum

import numpy as np
from dataclasses import dataclass, field

from src.alpha_scheduler import AlphaScheduler, Constant
from src.backprojection import apply_filter
from src.filters import RamLak
from src.linear_operators import XrayOperator
from src.optimizer import Optimizer, OptimizationResult
from src.formulations import Formulation, LSFormulation
from src.utils import power_iteration

from challenge.utils import calculate_score, segment


class Method(StrEnum):
    gradient_descent = "gradient_descent",
    momentum = "momentum",
    sirt = "sirt",
    adam = "adam",
    optimized_gradient = "optimized_gradient",
    nesterov_momentum = "nesterov_momentum",
    fast_gradient = "fast_gradient",
    conjugate_gradient = "conjugate_gradient",
    proximal_gradient = "proximal_gradient",
    fast_proximal_gradient = "fast_proximal_gradient",
    optimized_proximal_gradient = "optimized_proximal_gradient",
    admm = "admm",
    subgradient = "subgradient"


methods_with_operators = [Method.sirt, Method.conjugate_gradient, Method.admm]


@dataclass
class CTReconstructionProblem:
    phantom: str = "a"
    difficulty: int = 7
    arc: int = 360


@dataclass
class CTReconstructionSetup:
    formulation: Formulation = None
    formulation_params: dict = field(default_factory=dict)
    alpha_scheduler: AlphaScheduler = None
    alpha_scheduler_params: dict = field(default_factory=dict)
    method: Method = None
    method_params: dict = field(default_factory=dict)


@dataclass
class CTReconstructionResults:
    score: int = None
    x: np.ndarray = None
    i: int = None
    loss_history: np.ndarray = None
    score_history: np.ndarray = None
    n_iter: int = None
    converged: bool = False
    early_stopping: bool = False


@dataclass
class CTReconstructionRun:
    problem: CTReconstructionProblem = None
    setup: CTReconstructionSetup = None
    results: CTReconstructionResults = None


class CTReconstructionRunner:
    def __init__(self,
                 sinogram: np.ndarray,
                 operator: XrayOperator,
                 ground_truth: np.ndarray = None,
                 x0: np.ndarray = None,
                 alpha: float = None,
                 max_iter: int = 100,
                 tolerance: float = 1e-6,
                 early_stopping: bool = False,
                 patience: int = 20,
                 restarting: bool = False,
                 restart_frequency: int = -1,
                 ):
        self.segmented_ground_truth = segment(ground_truth)
        self.optimizer = Optimizer(
            formulation=LSFormulation(operator, sinogram.flatten()),
            x0=x0 if x0 is not None else operator.applyAdjoint(apply_filter(sinogram, RamLak())).flatten(),
            alpha=alpha if alpha is not None else 1 / power_iteration(operator.T @ operator, 100),
            max_iter=max_iter,
            tol=tolerance,
            early_stopping=early_stopping,
            patience=patience,
            restarting=restarting,
            restart_frequency=restart_frequency,
            verbose=True
        )
        self.patience = patience

    def _score(self, reconstruction):
        return calculate_score(segment(reconstruction.reshape(self.segmented_ground_truth.shape)),
                               self.segmented_ground_truth)

    def _pack_results(self, result: OptimizationResult, conf: CTReconstructionRun) -> CTReconstructionRun:
        conf.results = CTReconstructionResults()
        conf.results.score_history = [self._score(x) for x in result.x_history]
        conf.results.score = np.max(conf.results.score_history)
        conf.results.x = result.x_history[np.argmax(conf.results.score_history)]
        conf.results.i = np.argmax(conf.results.score_history)
        conf.results.loss_history = result.loss_history
        conf.results.n_iter = result.n_iter
        conf.results.converged = result.converged
        conf.results.early_stopping = result.early_stopping

        return conf

    @staticmethod
    def _serialize_setup(conf: CTReconstructionRun) -> CTReconstructionRun:
        conf.setup.formulation_params = vars(conf.setup.formulation).get("beta", None)
        conf.setup.formulation = conf.setup.formulation.__class__.__name__
        if conf.setup.alpha_scheduler is not None:
            conf.setup.alpha_scheduler_params = vars(conf.setup.alpha_scheduler).get("option", None)
            conf.setup.alpha_scheduler = conf.setup.alpha_scheduler.__class__.__name__
        else:
            conf.setup.alpha_scheduler_params = None
            conf.setup.alpha_scheduler = Constant.__name__
        conf.setup.method = conf.setup.method.value
        if conf.setup.method in methods_with_operators:
            conf.setup.method_params = {}
        return conf

    def run(self, conf: CTReconstructionRun) -> CTReconstructionRun:
        self.optimizer.patience = self.patience
        self.optimizer.formulation = conf.setup.formulation
        if conf.setup.alpha_scheduler is not None:
            self.optimizer.alpha_scheduler = conf.setup.alpha_scheduler
        else:
            self.optimizer.alpha_scheduler = Constant()

        result = getattr(self.optimizer, conf.setup.method)(**conf.setup.method_params)

        conf = self._pack_results(result, conf)

        conf = self._serialize_setup(conf)

        return conf
