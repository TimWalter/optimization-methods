from typing import List

import numpy as np
import matplotlib.pyplot as plt

from src.optimizer import OptimizationResult

from challenge.utils import calculate_score, segment


def plot_scores(results: List[OptimizationResult], labels: List[str], ground_truth: np.ndarray, title: str,
                filename: str = None):
    """Plot scores

    Args:
        results (List[OptimizationResult]): List of optimization results
        labels (List[str]): List of labels
        ground_truth (np.ndarray): Ground truth (Already segmented)
        title (str): Title of the plot
        filename (str, optional): Filename to save the plot to. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for result, label in zip(results, labels):
        data = [calculate_score(segment(x.reshape(ground_truth.shape)), ground_truth) for x in result.x_history]
        ax.plot(data, label=label + f" ({np.max(data):.2f})", alpha=0.5)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')


def plot_reconstructions(results: List[OptimizationResult], titles: List[str], suptitle: str, shape=(512, 512),
                         filename: str = None, indices: List[int] = None):
    """Plot reconstructions

    Args:
        results (List[OptimizationResult]): List of optimization results
        titles (List[str]): List of titles
        suptitle (str): Suptitle of the plot
        shape (tuple, optional): Shape of the images. Defaults to (512, 512).
        filename (str, optional): Filename to save the plot to. Defaults to None.
        indices (List[int], optional): List of indices to plot. Defaults to the final iteration.
    """
    reconstructions = []
    if indices is None:
        for result in results:
            reconstructions.append(result.x.reshape(shape))
    else:
        for index, result in zip(indices, results):
            reconstructions.append(result.x_history[index].reshape(shape))

    nCols = np.min([5, len(reconstructions)])
    nRows = int(np.ceil(len(reconstructions) / nCols))
    fig, axes = plt.subplots(nRows, nCols, figsize=(10, 5))
    for i, (ax, reconstruction, title) in enumerate(zip(axes.flatten(), reconstructions, titles)):
        ax.imshow(reconstruction, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    for i in range(len(reconstructions), nCols * nRows):
        axes.flatten()[i].axis('off')

    fig.suptitle(suptitle)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')


def plot_convergence(results: List[OptimizationResult], labels: List[str], title: str, filename: str = None):
    """Plot convergence

    Args:
        results (List[OptimizationResult]): List of optimization results
        labels (List[str]): List of labels
        title (str): Title of the plot
        filename (str, optional): Filename to save the plot to. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for result, label in zip(results, labels):
        ax.semilogy(result.loss_history, label=label, alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective function')
    ax.legend()
    ax.set_title(title)
    if filename:
        plt.savefig(filename, bbox_inches='tight')
