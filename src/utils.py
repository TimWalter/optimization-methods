import numpy as np
from scipy.sparse.linalg import LinearOperator

from src.linear_operators import StackedOperator
from src.proximal_operators import SeparableSumOperator
from src.formulations import TVRegularizationFormulation
from challenge.utils import calculate_score, segment

from tifffile import imwrite


def power_iteration(a: LinearOperator, num_iterations: int):
    """Power iteration algorithm

    Args:
        a (LinearOperator): Linear operator
        num_iterations (int): Number of iterations

    Returns:
        np.ndarray: Eigenvalue
    """
    b_k = np.random.rand(a.shape[1])
    b_k /= np.linalg.norm(b_k)

    for _ in range(num_iterations):
        b_k1 = a @ b_k
        b_k = b_k1 / np.linalg.norm(b_k1)

    eigenvalues = b_k.T @ (a @ b_k / (b_k.T @ b_k))
    return eigenvalues


def scale_landweber(alpha: float, a: LinearOperator, num_iterations: int):
    """Scale Landweber alpha

    Args:
        alpha (float): Alpha from [0, 1] that decides which step size in [0, 2/sigma_max**2] to use
        a (LinearOperator): Linear operator
        num_iterations (int): Number of iterations

    Returns:
        float: Landweber alpha
    """
    return alpha * 2 / power_iteration(a.T @ a, num_iterations)


def add_gaussian_noise(img: np.ndarray, sigma: float):
    """Add gaussian noise to an image

    Args:
        img (np.ndarray): Image
        sigma (float): Standard deviation of the gaussian noise

    Returns:
        np.ndarray: Image with added gaussian noise
    """
    return img + np.random.normal(0, sigma, img.shape)


def add_poisson_noise(img: np.ndarray, lam: float):
    """Add poisson noise to an image

    Args:
        img (np.ndarray): Image
        lam (float): Lambda of the poission noise

    Returns:
        np.ndarray: Image with added poission noise
    """
    return img + np.random.poisson(lam, img.shape)


def add_salt_and_pepper_noise(img: np.ndarray, prob: float):
    """Add salt and pepper noise to an image

    Args:
        img (np.ndarray): Image
        prob (float): Probability of the salt and pepper noise

    Returns:
        np.ndarray: Image with added salt and pepper noise
    """
    output = np.copy(img)
    output[np.random.rand(img.shape[0], img.shape[1]) < prob / 2] = np.min(img)
    output[np.random.rand(img.shape[0], img.shape[1]) < prob / 2] = np.max(img)

    return output


def calculate_best_score(result, ground_truth: np.ndarray):
    """Calculate the best score in a given run

    Args:
        result (np.ndarray): Result of the run
        ground_truth (np.ndarray): Ground truth (already segmented)

    Returns:
        tuple: Best score, position of the best score
    """
    scores = [calculate_score(segment(x.reshape(ground_truth.shape)), ground_truth) for x in result.x_history]
    return max(scores), np.argmax(scores)


# Precompute the distances from the center
y_coords, x_coords = np.ogrid[:512, :512]
distances_squared = (x_coords - 256) ** 2 + (y_coords - 256) ** 2


def best_crop(reconstruction: np.ndarray, flattened=False):
    """Find the best crop of the sinogram

    Args:
        reconstruction (np.ndarray): Reconstruction
        flattened (bool, optional): Whether the reconstruction is flattened. Defaults to False.

    Returns:
        Cropped reconstruction
    """
    if flattened:
        reconstruction = reconstruction.reshape((512, 512))

    gradient = np.sum(np.gradient(reconstruction), axis=0)
    x, y = np.unravel_index(np.argmax(gradient, axis=None), gradient.shape)

    radius_squared = (x - 256) ** 2 + (y - 256) ** 2

    mask = distances_squared >= radius_squared
    reconstruction[mask] = 0
    return reconstruction.flatten() if flattened else reconstruction


def save_run(run):
    path = f"/home/timwalter/CSE/inverse/aomip-tim-winter/data/submissions/{run.problem.phantom}/"
    imwrite(path + f"{run.problem.difficulty}_{run.problem.arc}_{run.setup.formulation}_{run.setup.method}.tif",
            run.results.x.reshape((512, 512)))
