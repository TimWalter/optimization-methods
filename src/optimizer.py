import numpy as np
from typing import Callable
from dataclasses import dataclass
from scipy.sparse.linalg import LinearOperator

from src.alpha_scheduler import AlphaScheduler, Constant
from src.linear_operators import StackedOperator
from src.formulations import Formulation, DfFormulation, PfFormulation, LSFormulation
from src.utils import best_crop

@dataclass
class OptimizationResult:
    x: np.ndarray = None
    x_history: np.ndarray = None
    loss_history: np.ndarray = None
    n_iter: int = 0
    converged: bool = False
    early_stopping: bool = False
    optimizer_configuration: dict = None


class Optimizer:
    def __init__(self, formulation: Formulation, x0: np.ndarray, alpha: float = 0.01,
                 alpha_scheduler: AlphaScheduler = Constant(), max_iter: int = 1000, tol: float = 1e-5,
                 early_stopping: bool = True, patience: int = 15, restarting: bool = False,
                 restart_frequency: int = 100,
                 verbose: bool = False):
        """Base class for optimizers

        Args:
            formulation (Formulation): Problem formulation
            x0 (np.ndarray): Initial guess
            alpha (float, optional): Step size (learning rate). Defaults to 0.01.
            alpha_scheduler (AlphaScheduler, optional): Alpha scheduler. Defaults to Constant().
            max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
            tol (float, optional): Convergence threshold. Defaults to 1e-5.
            early_stopping (bool, optional): Whether to stop early if no improvement is made. Defaults to True.
            patience (int, optional): Number of iterations to wait before stopping early. Defaults to 15.
            restarting (bool, optional): Whether to restart the optimizer after k iterations. Defaults to False.
            restart_frequency (int, optional): Number of iterations after which to restart the optimizer. Defaults to 100.
                                            -1 indicates adaptive restarting in case of increasing loss.
            verbose (bool, optional): Whether to return x and loss history. Defaults to False.
        """
        self.formulation = formulation
        self.x0 = x0
        self.alpha = alpha
        self.alpha_scheduler = alpha_scheduler
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.patience = patience
        self.restarting = restarting
        self.reset_frequency = restart_frequency
        self.verbose = verbose

        #if verbose:
            #self.__disable_unfitting_algorithms()

        self.state = None

    def __disable_unfitting_algorithms(self):
        """Disable algorithms that are not compatible with the problem formulation"""
        if not isinstance(self.formulation, LSFormulation):
            self.sirt = None
            self.conjugate_gradient = None
        if not isinstance(self.formulation, DfFormulation):
            self.gradient_descent = None
            self.momentum = None
            self.adam = None
            self.optimized_gradient = None
            self.nesterov_momentum = None
            self.fast_gradient = None
        if not isinstance(self.formulation, PfFormulation):
            self.admm = None
            self.proximal_gradient = None
            self.fast_proximal_gradient = None
            self.optimized_proximal_gradient = None
        if getattr(self.formulation, 'sub_df', None) is None:
            self.subgradient = None

    def __optimize(self, step: Callable, reset: Callable) -> OptimizationResult:
        """Optimize function

        Args:
            step (Callable): Step function
            reset (Callable): Reset function

        Returns:
            np.ndarray: Minimizer
        """
        result = OptimizationResult()
        result.x_history = [self.x0]
        result.optimizer_configuration = vars(self)

        if self.verbose or self.early_stopping:
            x_min = self.x0
            loss_min = self.formulation.f(self.x0)
            result.loss_history = [loss_min]
            patience_left = self.patience

        x = self.x0
        alpha = self.alpha
        for i in range(self.max_iter):
            x_new = step(x, alpha, i)
            result.x_history.append(x_new)
            if np.linalg.norm(x_new - x) < self.tol:
                print(f"Converged after {i} iterations")
                result.converged = True
                break
            if self.verbose or self.early_stopping:
                loss_new = self.formulation.f(x_new)
                result.loss_history.append(loss_new)
                if loss_new > 10e10:
                    print(f"Loss exploded after {i} iterations (loss: {loss_new})")
                    break
                # Reset patience if improvement is made
                if self.early_stopping:
                    if loss_new < loss_min:
                        x_min = x_new
                        loss_min = loss_new
                        patience_left = self.patience
                    else:
                        patience_left -= 1
                        if self.restarting and self.reset_frequency == -1:
                            alpha = self.alpha
                            reset(x)
                        if patience_left == 0:
                            x = x_min
                            print(f"Early stopping after {i + 1 - self.patience} iterations")
                            result.early_stopping = True
                            break
            alpha = self.alpha_scheduler(alpha, x_new, x, i)
            x = x_new
            if self.restarting and self.reset_frequency != -1 and (i + 1) % self.reset_frequency == 0:
                alpha = self.alpha
                reset(x)
        if self.verbose:
            print(f"Iterated {i + 1} times \n Final loss: {self.formulation.f(x)}")
        result.x = x
        result.n_iter = i + 1
        return result

    def gradient_descent(self) -> np.ndarray | OptimizationResult:
        """Gradient descent algorithm

        Returns:
            np.ndarray: Minimizer
        """

        @dataclass
        class GradientDescentState:
            pass

        def reset(x: np.ndarray):
            pass

        def gradient_descent_step(x: np.ndarray, alpha: float, _: int) -> np.ndarray:
            return x - alpha * self.formulation.df(x)

        self.state = GradientDescentState()
        return self.__optimize(gradient_descent_step, reset)

    def momentum(self, beta: float = 0.9) -> OptimizationResult:
        """Momentum algorithm

        Args:
            beta (float, optional): Momentum decay rate. Defaults to 0.9.

        Returns:
            np.ndarray: Minimizer
        """

        @dataclass
        class MomentumState:
            momentum: np.ndarray

        def reset(_: np.ndarray):
            self.state.momentum = np.zeros_like(self.x0)

        def momentum_step(x: np.ndarray, alpha: float, _: int) -> np.ndarray:
            self.state.momentum = beta * self.state.momentum - alpha * self.formulation.df(x)
            return x + self.state.momentum

        self.state = MomentumState(momentum=np.zeros_like(self.x0))
        return self.__optimize(momentum_step, reset)

    def sirt(self, a: LinearOperator, b: np.ndarray) -> OptimizationResult:
        """ Simultaneous Iterative Reconstruction Technique (SIRT)

        Args:
            a (LinearOperator): Linear operator
            b (np.ndarray): Right-hand side (sinogram)

        Returns:
            np.ndarray: Minimizer
        """

        @dataclass
        class SIRTState:
            inv_row_sum: np.ndarray
            inv_col_sum: np.ndarray

        def reset(_: np.ndarray):
            pass

        def sirt_step(x: np.ndarray, alpha: float, _: int) -> np.ndarray:
            return x + alpha * self.state.inv_col_sum * (a.T @ (self.state.inv_row_sum * (a @ x - b)))

        self.state = SIRTState(1 / ((a.T @ np.ones(a.shape[0])) + np.finfo(float).eps),
                               1 / ((a @ np.ones(a.shape[1])) + np.finfo(float).eps))
        return self.__optimize(sirt_step, reset)

    def adam(self, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> OptimizationResult:
        """Adam algorithm

        Args:
            beta1 (float, optional): Exponential decay rate for first moment estimates. Defaults to 0.9.
            beta2 (float, optional): Exponential decay rate for second moment estimates. Defaults to 0.999.
            epsilon (float, optional): Small constant to avoid division by zero. Defaults to 1e-8.

        Returns:
            np.ndarray: Minimizer
        """

        @dataclass
        class AdamState:
            m: np.ndarray
            v: np.ndarray

        def reset(_: np.ndarray):
            self.state.m = np.zeros_like(self.x0)
            self.state.v = np.zeros_like(self.x0)

        def adam_step(x: np.ndarray, alpha: float, _: int) -> np.ndarray:
            g = self.formulation.df(x)
            self.state.m = beta1 * self.state.m + (1 - beta1) * g
            self.state.v = beta2 * self.state.v + (1 - beta2) * g ** 2
            m_hat = self.state.m / (1 - beta1)
            v_hat = self.state.v / (1 - beta2)
            return x - alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        self.state = AdamState(m=np.zeros_like(self.x0), v=np.zeros_like(self.x0))
        return self.__optimize(adam_step, reset)

    def optimized_gradient(self) -> OptimizationResult:
        """Optimized gradient algorithm

        Returns:
            np.ndarray: Minimizer
        """

        @dataclass
        class OptimizedGradientState:
            gradient_step: np.ndarray
            momentum: np.ndarray

        def reset(x: np.ndarray):
            self.state.gradient_step = np.zeros_like(x)
            self.state.momentum = np.ones_like(x)

        def optimized_gradient_step(x: np.ndarray, alpha: float, i: int) -> np.ndarray:
            gradient_step_new = x - alpha * self.formulation.df(x)
            if i != self.max_iter - 1:
                momentum_new = 0.5 + np.sqrt(0.25 + self.state.momentum ** 2)
            else:
                momentum_new = 0.5 + np.sqrt(0.25 + 2 * self.state.momentum ** 2)
            x_new = gradient_step_new + (self.state.momentum - 1) / momentum_new * (
                    gradient_step_new - self.state.gradient_step) + self.state.momentum / momentum_new * (
                            gradient_step_new - x)
            self.state.gradient_step = gradient_step_new
            self.state.momentum = momentum_new
            return x_new

        self.state = OptimizedGradientState(gradient_step=np.zeros_like(self.x0), momentum=np.ones_like(self.x0))
        return self.__optimize(optimized_gradient_step, reset)

    def nesterov_momentum(self, beta: float = 0.9) -> OptimizationResult:
        """Nesterov momentum algorithm

        Args:
            beta (float, optional): Exponential decay rate for momentum. Defaults to 0.9.

        Returns:
            np.ndarray: Minimizer
        """

        @dataclass
        class NesterovMomentumState:
            momentum: np.ndarray

        def reset(x: np.ndarray):
            self.state.momentum = np.zeros_like(x)

        def nesterov_momentum_step(x: np.ndarray, alpha: float, _: int) -> np.ndarray:
            self.state.momentum = beta * self.state.momentum - alpha * self.formulation.df(
                x + beta * self.state.momentum)
            return x + self.state.momentum

        self.state = NesterovMomentumState(momentum=np.zeros_like(self.x0))
        return self.__optimize(nesterov_momentum_step, reset)

    def fast_gradient(self) -> OptimizationResult:
        """Nesterov's fast gradient


        Returns:
            np.ndarray: Minimizer
        """

        @dataclass
        class FastGradientState:
            momentum: np.ndarray
            beta: float

        def reset(x: np.ndarray):
            self.state.momentum = x

        def fast_gradient_step(x: np.ndarray, alpha: float, _: int) -> np.ndarray:
            momentum_new = x - alpha * self.formulation.df(x)
            beta_new = 0.5 + np.sqrt(0.25 + self.state.beta ** 2)
            x_new = momentum_new + (self.state.beta - 1) / beta_new * (momentum_new - self.state.momentum)
            self.state.momentum = momentum_new
            self.state.beta = beta_new
            return x_new

        self.state = FastGradientState(momentum=self.x0, beta=1)
        return self.__optimize(fast_gradient_step, reset)

    def conjugate_gradient(self, a: LinearOperator, b: np.ndarray) -> OptimizationResult:
        """Conjugate gradient algorithm

        Args:
            a (LinearOperator): Symmetric positive-definite linear operator
            b (np.ndarray): Right-hand side vector

        Returns:
            np.ndarray: Minimizer
        """

        @dataclass
        class ConjugateGradientState:
            residual: np.ndarray
            step_direction: np.ndarray

        def reset(x: np.ndarray):
            self.state.residual = b - a @ x
            self.state.step_direction = self.state.residual

        def conjugate_gradient_step(x: np.ndarray, _: float, __: int) -> np.ndarray:
            alpha = self.state.residual.T @ self.state.residual / (
                    self.state.step_direction.T @ (a @ self.state.step_direction) + np.finfo(float).eps)
            x_new = x + alpha * self.state.step_direction
            residual_new = self.state.residual - alpha * (a @ self.state.step_direction)
            beta = residual_new.T @ residual_new / (
                    self.state.residual.T @ self.state.residual + np.finfo(float).eps)
            self.state.step_direction = residual_new + beta * self.state.step_direction
            self.state.residual = residual_new
            return x_new

        self.state = ConjugateGradientState(residual=b - a @ self.x0, step_direction=b - a @ self.x0)
        return self.__optimize(conjugate_gradient_step, reset)

    def proximal_gradient(self):
        """Proximal gradient algorithm


        Returns:
            np.ndarray: Minimizer
        """

        @dataclass
        class ProximalGradientState:
            pass

        def reset(_: np.ndarray):
            pass

        def proximal_gradient_step(x: np.ndarray, alpha: float, _: int) -> np.ndarray:
            gradient_step = x - alpha * self.formulation.partial_df(x)
            return self.formulation.proximal_operator(gradient_step, alpha)

        self.state = ProximalGradientState()
        return self.__optimize(proximal_gradient_step, reset)

    def fast_proximal_gradient(self, momentum_choice: int = 0):
        """Fast proximal gradient algorithm

        Args:
            momentum_choice (int, optional): Momentum choice 0 or 1. Defaults to 0.

        Returns:
            np.ndarray: Minimizer
        """

        @dataclass
        class FastProximalGradientState:
            z: np.ndarray
            momentum_choice: int
            t: float = 1

        def reset(_: np.ndarray):
            self.state.t = 1

        def fast_proximal_gradient_step(x: np.ndarray, alpha: float, i: int) -> np.ndarray:
            gradient_step = x - alpha * self.formulation.partial_df(x)
            z_new = self.formulation.proximal_operator(gradient_step, alpha)
            if self.state.momentum_choice == 0:
                momentum_alpha = i / (i + 1)
            elif self.state.momentum_choice == 1:
                t_new = 0.5 * (1 + np.sqrt(1 + 4 * self.state.t ** 2))
                momentum_alpha = (self.state.t - 1) / t_new
                self.state.t = t_new
            else:
                raise ValueError("Momentum choice must be 0 or 1")
            x_new = z_new + momentum_alpha * (z_new - self.state.z)
            self.state.z = z_new
            return x_new

        self.state = FastProximalGradientState(z=self.x0, momentum_choice=momentum_choice)
        return self.__optimize(fast_proximal_gradient_step, reset)

    def optimized_proximal_gradient(self):
        """Optimized proximal gradient algorithm


        Returns:
            np.ndarray: Minimizer
        """

        @dataclass
        class OptimizedProximalGradientState:
            gradient_step: np.ndarray
            full_step: np.ndarray
            theta: float = 1
            gamma: float = 1

        def reset(x: np.ndarray):
            self.state.gradient_step = x
            self.full_step = x
            self.state.theta = 1
            self.state.gamma = 1

        def optimized_proximal_gradient_step(x: np.ndarray, alpha: float, i: int) -> np.ndarray:
            if i != self.max_iter - 1:
                theta_new = 0.5 + np.sqrt(0.25 + self.state.theta ** 2)
            else:
                theta_new = 0.5 + np.sqrt(0.25 + 2 * self.state.theta ** 2)

            gamma_new = alpha * (2 * self.state.theta + theta_new - 1) / theta_new

            gradient_step_new = x - alpha * self.formulation.partial_df(x)
            nesterov_step = (self.state.theta - 1) / theta_new * (gradient_step_new - self.state.gradient_step)
            ogm_step = self.state.theta / theta_new * (gradient_step_new - x)
            pogm_step = alpha * (self.state.theta - 1) / theta_new / self.state.gamma * (self.state.full_step - x)
            full_step_new = gradient_step_new + nesterov_step + ogm_step + pogm_step

            x_new = self.formulation.proximal_operator(full_step_new, gamma_new)

            self.state.gradient_step = gradient_step_new
            self.state.full_step = full_step_new
            self.state.theta = theta_new
            self.state.gamma = gamma_new

            return x_new

        self.state = OptimizedProximalGradientState(gradient_step=self.x0, full_step=self.x0)
        return self.__optimize(optimized_proximal_gradient_step, reset)

    def admm(self, mu: float, tau: float):
        """Alternating Direction Method of Multipliers

        Args:
            mu (float): Sigma for proximal operator, makes up the step size together with tau
            tau (float): Sigma for proximal operator with linear operator, makes up the step size together with mu

        Returns:
            np.ndarray: Minimizer
        """

        @dataclass
        class ADMMState:
            mu: float
            tau: float
            z: np.ndarray
            u: np.ndarray

        def reset(x: np.ndarray):
            self.state.z = k @ x
            self.state.u = np.zeros_like(self.state.z)

        def admm_step(x: np.ndarray, _: float, __: int) -> np.ndarray:
            x_new = self.formulation.proximal_operator(
                x - self.state.mu / self.state.tau * k.T @ (k @ x - self.state.z + self.state.u), self.state.mu)

            #TEST
            x_new = best_crop(x_new, True)

            z_new = self.formulation.proximal_operator_lo(k @ x_new + self.state.u, self.state.tau)
            u_new = self.state.u + k @ x_new - z_new

            self.state.z = z_new
            self.state.u = u_new
            return x_new

        k = StackedOperator(self.formulation.get_linear_operators_in_pol())

        self.state = ADMMState(mu=mu, tau=tau, z=self.formulation.proximal_operator_lo(k @ self.x0, tau),
                               u=np.zeros(k.shape[0]))
        return self.__optimize(admm_step, reset)

    def subgradient(self):
        """Subgradient method

        Returns:
            np.ndarray: Minimizer
        """

        @dataclass
        class SubgradientState:
            pass

        def reset(_: np.ndarray):
            pass

        def subgradient_step(x: np.ndarray, alpha: float, __: int) -> np.ndarray:
            return x - alpha * self.sub_df(x)

        self.state = SubgradientState()
        return self.__optimize(subgradient_step, reset)
