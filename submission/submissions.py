import numpy as np

from src.phantoms import helsinki_challenge
from src.formulations import LSFormulation, TikhonovFormulation, LassoFormulation, LSDfRFormulation, \
    ElasticNetFormulation, TVRegularizationFormulation, DfFormulation, PfFormulation
from src.alpha_scheduler import Constant, Decay, Backtracking, Wolfe, BarzilaiBorwein
from src.regularizer import HuberRegularizer, FairRegularizer

from src.runner import Method, CTReconstructionRun, CTReconstructionRunner, CTReconstructionProblem, \
    CTReconstructionSetup

from src.filters import RamLak, Cosine
from src.backprojection import apply_filter
from src.utils import best_crop, power_iteration

from src.linear_operators import CentralDifferences, StackedOperator

import pickle
from datetime import datetime
from os.path import isfile

best_fbp_filter = {
    "a": {
        1: {},
        2: {},
        3: {},
        4: {},
        5: {},
        6: {},
        7: {
            360: RamLak,
            90: Cosine,
            60: Cosine,
            30: RamLak,
        },
    },
    "b": {
        1: {},
        2: {},
        3: {},
        4: {},
        5: {},
        6: {},
        7: {},
    },
    "c": {
        1: {},
        2: {},
        3: {},
        4: {},
        5: {},
        6: {},
        7: {},
    },
}


def best_fbp(phantom, difficulty, arc, xray_operator, sinogram):
    """Run the predetermined FBP, which results in the best initial reconstruction

    Args:
        phantom (str): Phantom name
        difficulty (int): Difficulty
        arc (int): Arc
        xray_operator (XRayOperator): Xray transform
        sinogram (np.ndarray): Sinogram
     """
    x0 = xray_operator.applyAdjoint(apply_filter(sinogram, best_fbp_filter["a"][7][arc]()))
    if arc != 360:
        x0 = best_crop(x0)
    return x0.flatten()


def create_runs_1_methods_schedulers(phantom, difficulty, arc, xray_operator, sinogram, ground_truth):
    problem = CTReconstructionProblem(phantom, difficulty, arc)

    formulations = [
        LSFormulation(xray_operator, sinogram),
        TikhonovFormulation(xray_operator, sinogram),
        LSDfRFormulation(xray_operator, sinogram, regularizer=[HuberRegularizer(1e-2)]),
        LSDfRFormulation(xray_operator, sinogram, regularizer=[FairRegularizer(1e-2)]),
        LassoFormulation(xray_operator, sinogram, 1e-2),
        ElasticNetFormulation(xray_operator, sinogram, [0.5, 1e-2]),
        TVRegularizationFormulation(xray_operator, sinogram, 1, CentralDifferences(ground_truth.shape), True,
                                    (1024, 512)),
    ]

    setups = []
    for formulation in formulations:
        if isinstance(formulation, DfFormulation):
            setups += [
                CTReconstructionSetup(
                    formulation=formulation,
                    method=Method.gradient_descent,
                    alpha_scheduler=BarzilaiBorwein(formulation.f, formulation.df, "short"),
                ),
                CTReconstructionSetup(
                    formulation=formulation,
                    method=Method.adam,
                ),
                CTReconstructionSetup(
                    formulation=formulation,
                    method=Method.optimized_gradient,
                ),
                CTReconstructionSetup(
                    formulation=formulation,
                    method=Method.conjugate_gradient,
                    method_params={"a": xray_operator.T @ xray_operator, "b": xray_operator.T @ sinogram},
                ),
            ]
        elif isinstance(formulation, PfFormulation):
            if not isinstance(formulation, TVRegularizationFormulation):
                setups += [
                    CTReconstructionSetup(
                        formulation=formulation,
                        method=Method.proximal_gradient,
                    ),
                    CTReconstructionSetup(
                        formulation=formulation,
                        method=Method.optimized_proximal_gradient,
                    ),
                ]
            k = StackedOperator(formulation.get_linear_operators_in_pol())
            mu = 0.8 / power_iteration(k.T @ k, 100)
            setups += [CTReconstructionSetup(
                formulation=formulation,
                method=Method.admm,
                method_params={"mu": mu, "tau": 100},
            ), ]
    runs = []
    for setup in setups:
        runs.append(CTReconstructionRun(problem, setup))

    return runs


def create_runs_2_formulations_parameters(phantom, difficulty, arc, xray_operator, sinogram, ground_truth):
    problem = CTReconstructionProblem(phantom, difficulty, arc)

    formulations = [LSFormulation(xray_operator, sinogram), ]

    for beta in np.logspace(-5, 0, 10):
        formulations += [
            TikhonovFormulation(xray_operator, sinogram, beta),
            LassoFormulation(xray_operator, sinogram, beta),
            TVRegularizationFormulation(xray_operator, sinogram, beta, CentralDifferences(ground_truth.shape), True,
                                        (1024, 512)),
        ]
        for delta in np.logspace(-3, -1, 3):
            formulations += [
                LSDfRFormulation(xray_operator, sinogram, [beta], regularizer=[HuberRegularizer(delta)]),
                LSDfRFormulation(xray_operator, sinogram, [beta], regularizer=[FairRegularizer(delta)]),
            ]
        for beta2 in np.logspace(-5, 0, 5):
            formulations += [
                ElasticNetFormulation(xray_operator, sinogram, [beta2, beta]),
            ]

    setups = []
    for formulation in formulations:
        if isinstance(formulation, TVRegularizationFormulation):
            k = StackedOperator(formulation.get_linear_operators_in_pol())
            mu = 0.8 / power_iteration(k.T @ k, 50)
            setups += [CTReconstructionSetup(
                formulation=formulation,
                method=Method.admm,
                method_params={"mu": mu, "tau": 100},
            ), ]
        elif isinstance(formulation, ElasticNetFormulation) or isinstance(formulation, LassoFormulation):
            k = StackedOperator(formulation.get_linear_operators_in_pol())
            mu = 0.8 / power_iteration(k.T @ k, 50)
            setups += [
                CTReconstructionSetup(
                    formulation=formulation,
                    method=Method.admm,
                    method_params={"mu": mu, "tau": 100},
                ),
                CTReconstructionSetup(
                    formulation=formulation,
                    method=Method.optimized_proximal_gradient,
                ),
            ]
        elif isinstance(formulation, TikhonovFormulation):
            setups += [
                CTReconstructionSetup(
                    formulation=formulation,
                    method=Method.gradient_descent,
                    alpha_scheduler=BarzilaiBorwein(formulation.f, formulation.df, "short"),
                ),
                CTReconstructionSetup(
                    formulation=formulation,
                    method=Method.conjugate_gradient,
                    method_params={"a": xray_operator.T @ xray_operator, "b": xray_operator.T @ sinogram},
                ),
            ]
        elif isinstance(formulation, LSDfRFormulation):
            setups += [
                CTReconstructionSetup(
                    formulation=formulation,
                    method=Method.optimized_gradient,
                ),
            ]
        elif isinstance(formulation, LSFormulation):
            setups += [
                CTReconstructionSetup(
                    formulation=formulation,
                    method=Method.gradient_descent,
                    alpha_scheduler=BarzilaiBorwein(formulation.f, formulation.df, "short"),
                ),
            ]

    runs = []
    for setup in setups:
        runs.append(CTReconstructionRun(problem, setup))

    return runs


def create_runs_3_formulations_parameters_refined(phantom, difficulty, arc, xray_operator, sinogram, ground_truth):
    problem = CTReconstructionProblem(phantom, difficulty, arc)

    if arc == 90 or arc == 60:
        formulations = []
        for beta in np.logspace(-1, 3, 10):
            formulations += [
                LSDfRFormulation(xray_operator, sinogram, [beta], regularizer=[HuberRegularizer(1e-2)]),
            ]
    elif arc == 30:
        formulations = []
        for beta in np.logspace(-1, 8, 10):
            for beta2 in np.logspace(-2, 0, 8):
                formulations += [
                    ElasticNetFormulation(xray_operator, sinogram, [beta, beta2]),
                ]
    else:
        formulations = []

    setups = []
    for formulation in formulations:
        if isinstance(formulation, ElasticNetFormulation) or isinstance(formulation, TVRegularizationFormulation):
            k = StackedOperator(formulation.get_linear_operators_in_pol())
            mu = 0.8 / power_iteration(k.T @ k, 50)
            setups += [
                CTReconstructionSetup(
                    formulation=formulation,
                    method=Method.admm,
                    method_params={"mu": mu, "tau": 100},
                )
            ]
        elif isinstance(formulation, LSDfRFormulation):
            setups += [
                CTReconstructionSetup(
                    formulation=formulation,
                    method=Method.optimized_gradient,
                ),
            ]
    runs = []
    for setup in setups:
        runs.append(CTReconstructionRun(problem, setup))

    return runs


def create_runs_final(phantom, difficulty, arc, xray_operator, sinogram, ground_truth):
    problem = CTReconstructionProblem(phantom, difficulty, arc)
    if arc == 360:
        formulations = [
            LSDfRFormulation(xray_operator, sinogram, [0.0215], regularizer=[HuberRegularizer(1e-2)]),
        ]
    elif arc == 90:
        formulations = [
            LassoFormulation(xray_operator, sinogram, 0.7742),
            LSDfRFormulation(xray_operator, sinogram, [2.154], regularizer=[HuberRegularizer(1e-2)]),
        ]
    elif arc == 60:
        formulations = [
            LassoFormulation(xray_operator, sinogram, 0.7742),
            LSDfRFormulation(xray_operator, sinogram, [0.7742], regularizer=[HuberRegularizer(1e-2)]),
        ]
    elif arc == 30:
        formulations = [
            LassoFormulation(xray_operator, sinogram, 0.00004641),
            TVRegularizationFormulation(xray_operator, sinogram, 0.0215, CentralDifferences(ground_truth.shape),
                                        True, (1024, 512)),
            ElasticNetFormulation(xray_operator, sinogram, [10, 1]),
        ]
    else:
        formulations = []

    setups = []
    for formulation in formulations:
        if isinstance(formulation, TVRegularizationFormulation) or isinstance(formulation, ElasticNetFormulation):
            k = StackedOperator(formulation.get_linear_operators_in_pol())
            mu = 0.8 / power_iteration(k.T @ k, 50)
            setups += [CTReconstructionSetup(
                formulation=formulation,
                method=Method.admm,
                method_params={"mu": mu, "tau": 100},
            ), ]
        elif isinstance(formulation, LassoFormulation):
            setups += [
                CTReconstructionSetup(
                    formulation=formulation,
                    method=Method.optimized_proximal_gradient,
                ),
            ]
        elif isinstance(formulation, LSDfRFormulation):
            setups += [
                CTReconstructionSetup(
                    formulation=formulation,
                    method=Method.optimized_gradient,
                ),
            ]
    runs = []
    for setup in setups:
        runs.append(CTReconstructionRun(problem, setup))

    return runs


def sweep_problem(phantom, difficulty, arc, name):
    print(f"[{datetime.now()}] Setting up {name}")
    path = "../../data/submissions/run_final/"
    print(f"[{datetime.now()}] {name}: Load data")
    sinogram, xray_operator, ground_truth = helsinki_challenge(phantom, difficulty, 0, arc)

    print(f"[{datetime.now()}] {name}: Setup runner")
    alpha = 0.5 / power_iteration(xray_operator.T @ xray_operator, 100)
    x0 = best_fbp(phantom, difficulty, arc, xray_operator, sinogram)

    sinogram = sinogram.flatten()

    runner = CTReconstructionRunner(
        sinogram,
        xray_operator,
        ground_truth,
        x0,
        alpha,
        1000,
        0,
        True,
        40,
    )
    print(f"[{datetime.now()}] {name}: Creating runs")
    runs = create_runs_final(phantom, difficulty, arc, xray_operator, sinogram, ground_truth)

    print(f"[{datetime.now()}] {name}: Starting runs")
    for i, run in enumerate(runs):
        filename = path + f"{name}_{i + 1}.pkl"
        if not isfile(filename):
            print(f"[{datetime.now()}] {name}: Running {i + 1}/{len(runs)}")
            result = runner.run(run)
            with open(filename, "wb") as f:
                pickle.dump(result, f)
        else:
            print(f"[{datetime.now()}] {name}: Skipping {i + 1}/{len(runs)}")


if __name__ == "__main__":
    for phantom in ["a", "b", "c"]:
        for difficulty in range(1, 8):
            for arc in [360, 90, 60, 30]:
                sweep_problem(phantom, difficulty, arc, f"{phantom}_{difficulty}_{arc}")
