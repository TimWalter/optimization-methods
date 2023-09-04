import numpy as np
from src.linear_operators import XrayOperator
from src.preprocessing import pad_sinogram
from src.filters import Filter


def cosine_weighting(sinogram: np.ndarray, s2c: float, c2d: float) -> np.ndarray:
    """Apply cosine weighting to sinogram to convert cone beams to parallel beams

    Args:
        sinogram (np.ndarray): Input sinogram
        s2c (float): Source to center distance
        c2d (float): Center to detector distance

    Returns:
        np.ndarray: Weighted sinogram
    """
    d = s2c + c2d
    u = (np.arange(sinogram.shape[0]) + 0.5) - sinogram.shape[0] / 2
    if sinogram.ndim == 3:
        v = (np.arange(sinogram.shape[1]) + 0.5) - sinogram.shape[1] / 2
        w = d / np.sqrt((d * d) + np.power(u, 2) + np.power(v, 2))
    elif sinogram.ndim == 2:
        w = d / np.sqrt((d * d) + np.power(u, 2))
    else:
        raise ValueError("Sinogram must be 2D or 3D")

    return sinogram * w[..., np.newaxis]


def apply_filter(sinogram: np.ndarray, filter_function: Filter) -> np.ndarray:
    """Apply filter to sinogram in Fourier space

    Args:
        sinogram (np.ndarray): Input sinogram
        filter_function (Filter): Filter function

    Returns:
        np.ndarray: Filtered sinogram
    """
    size = sinogram.shape[:-1]

    filter_vectors = [filter_function(s) for s in size]
    fourier_filter = filter_vectors[0]
    for i in range(1, len(filter_vectors)):
        fourier_filter = np.outer(fourier_filter, filter_vectors[i])

    fourier_filter = fourier_filter[..., np.newaxis]

    axes = tuple(range(len(sinogram.shape) - 1))
    fourier_sinogram = np.fft.fftn(sinogram, axes=axes)
    projection = np.fft.fftshift(fourier_sinogram, axes=-1) * np.fft.fftshift(fourier_filter, axes=axes)
    return np.real(np.fft.ifftn(np.fft.ifftshift(projection, axes=-1), axes=axes))


def filtered_backprojection(reconstruction_shape: np.ndarray, sinogram: np.ndarray, angles: np.ndarray,
                            filter_function: Filter, s2c: float = None, c2d: float = None,
                            parallel: bool = True) -> np.ndarray:
    """Apply filtered backprojection to sinogram

    Args:
        reconstruction_shape (np.ndarray): Expected size of the phantom (e.g. detector size)
        sinogram (np.ndarray): Input sinogram
        angles (np.ndarray): Projection angles
        filter_function (Filter): Filter function
        s2c (float): Source to center distance. Defaults to None.
        c2d (float): Center to detector distance. Defaults to None.
        parallel (bool, optional): Parallel beam or cone beam. Defaults to True.


    Returns:
        np.ndarray: Reconstructed phantom
    """
    if not parallel:
        sinogram = cosine_weighting(sinogram, s2c, c2d)
    projection_shape = sinogram.shape[:-1]
    desired_size = np.array(2 ** np.ceil(np.log2(projection_shape)), dtype=int)
    padding_width = np.array(desired_size - projection_shape, dtype=int)
    padded_sinogram = pad_sinogram(sinogram, padding_width, True)

    # Apply filter in Fourier domain
    filtered_sinogram = apply_filter(padded_sinogram, filter_function)

    # Trim padding again (probably unnecessary)
    if np.all(padding_width != 0):
        trim_axes = (*[slice(pw // 2, -pw // 2) for pw in padding_width], slice(None))
        filtered_sinogram = filtered_sinogram[*trim_axes]

    # Project back
    operator = XrayOperator(reconstruction_shape, filtered_sinogram.shape[:-1], angles, s2c, c2d)
    phantom = operator.applyAdjoint(filtered_sinogram)
    return phantom
