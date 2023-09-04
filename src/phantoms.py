import numpy as np
from tifffile import imread

import pyelsa as elsa

from src.linear_operators import XrayOperator
from challenge.utils import load_htc2022data


def shepp_logan(size: tuple, arc: int = 360, num_angles: int = 720) -> tuple[np.ndarray, XrayOperator, np.ndarray]:
    """Create a shepp logan phantom

    Args:
        size (tuple): Size of the phantom
        arc (int, optional): Arc. Defaults to 360.
        num_angles (int, optional): Number of angles. Default to 720.

    Returns:
        tuple[np.ndarray, XrayOperator, np.ndarray]: Sinogram, operator and ground truth
    """

    ground_truth = np.rot90(elsa.phantoms.modifiedSheppLogan(size), -1)
    operator = XrayOperator(size, [size[0]*2], np.linspace(0, arc, num_angles), size[0] * 100, size[0] * 2)
    sinogram = operator.apply(ground_truth)

    return sinogram, operator, ground_truth


def helsinki_challenge(phantom: str = "a", difficulty: int = 7, arc_start: int = 0, arc: int = 360) \
        -> tuple[np.ndarray, XrayOperator, np.ndarray]:
    """Load challenge dataset

    Args:
        phantom (str, optional): Phantom to load. Defaults to "a".
        difficulty (int, optional): Difficulty to load. Defaults to 7.
        arc_start (int, optional): Arc start. Defaults to 0.
        arc (int, optional): Arc. Defaults to 360.

    Returns:
        tuple[np.ndarray, XRayOperator, np.ndarray]: Sinogram, operator and ground truth
    """
    ground_truth = imread(
        f"/home/timwalter/CSE/inverse/src-tim-winter/data/2D/groundtruth/htc2022_0{difficulty}{phantom}_recon.tif")
    ground_truth = (ground_truth - np.min(ground_truth)) / (np.max(ground_truth) - np.min(ground_truth))

    path = "/home/timwalter/CSE/inverse/src-tim-winter/data/2D/htc2022_test_data/"
    filename = path + f"htc2022_0{difficulty}{phantom}_full.mat"

    sinogram, operator = load_htc2022data(filename, arc=arc, arcstart=arc_start)

    return sinogram, operator, ground_truth
