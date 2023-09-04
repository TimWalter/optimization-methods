import numpy as np
from numpy import ndarray


def flat_field_correction(measurement: np.ndarray, flat: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """Flat field correction

    Args:
        measurement (np.ndarray): Actual measurement with object
        flat (np.ndarray): Calibrating measurement without object
        dark (np.ndarray): Calibrating measurement without any radiation

    Returns:
        np.ndarray: Corrected measurement
    """
    return (measurement - dark) / (flat - dark)


def estimate_initial_intensity(x: np.ndarray) -> float:
    """Estimate initial intensity

    Args:
        x (np.ndarray): Transmission representation

    Returns:
        float: Initial intensity
    """
    w, h = x.shape
    scale = 20
    wc, hc = w // scale, h // scale
    corner_regions = np.array([x[:wc, :hc], x[:wc, -hc:], x[-wc:, :hc], x[-wc:, -hc:]])

    return np.mean(corner_regions)


def transmission_to_absorption(x: np.ndarray, i0: float = None) -> np.ndarray:
    """Convert transmission to absorption representation

    Args:
        x (np.ndarray): Transmission representation
        i0 (np.ndarray): Initial intensity

    Returns:
        np.ndarray: Absorption representation
    """
    if i0 is None:
        i0 = estimate_initial_intensity(x)
    return np.clip(-np.log(x / i0), 0, None)


def absorption_to_transmission(x: np.ndarray, i0: float) -> np.ndarray:
    """Convert absorption to transmission representation

        Args:
            x (np.ndarray): Absorption representation
            i0 (np.ndarray): Initial intensity

        Returns:
            np.ndarray: Transmission representation
        """
    return np.clip(i0 * np.exp(-x), None, i0)


def mean_pooling(x: np.ndarray, factor: int) -> np.ndarray:
    """Mean pooling

    Args:
        x (np.ndarray): Input array
        factor (int): Pooling factor

    Returns:
        np.ndarray: Output array

    Notes:
        if len(x.shape) == 1:
            w = x.shape[0]
            wc = w // factor
            return x.reshape(wc, factor).mean(axis=1)
        elif len(x.shape) == 2:
            w, h = x.shape
            wc, hc = w // factor, h // factor
            return x.reshape(wc, factor, hc, factor).mean(axis=(1, 3))
        elif len(x.shape) == 3:
            w, h, d = x.shape
            wc, hc, dc = w // factor, h // factor, d // factor
            return x.reshape(wc, factor, hc, factor, dc, factor).mean(axis=(1, 3, 5))
    """

    dim = len(x.shape)
    truncated_shape = np.array(x.shape) // factor
    new_shape = np.c_[truncated_shape, [factor] * len(x.shape)].flatten()
    return x.reshape(new_shape).mean(axis=tuple(np.arange(1, 2 * dim, 2, dtype=int)))


def shift_center_of_rotation(x: np.ndarray, shift: float) -> np.ndarray:
    """Shift center of rotation

    Args:
        x (np.ndarray): Sinogram
        shift (float): Shift

    Returns:
        np.ndarray: Shifted sinogram
    """
    return np.roll(x, int(shift), axis=-1)


def pad_sinogram(x: np.ndarray, pad_width: np.ndarray, symmetric=True) -> np.ndarray:
    """Pad sinogram

    Args:
        x (np.ndarray): Sinogram
        pad_width (int): Padding width
        symmetric (bool, optional): Whether to apply padding on both sides. Defaults to True.

    Returns:
        np.ndarray: Padded sinogram
    """
    if symmetric:
        pad_width //= 2
        padding_axes = (tuple([(pw, pw) for pw in pad_width]) + ((0, 0),))
        padded = np.pad(x, padding_axes, mode="symmetric")
    else:
        padding_axes = (tuple([(pw, 0) for pw in pad_width]) + ((0, 0),))
        padded = np.pad(x, padding_axes, mode="symmetric")
    return padded


def sinogram_slice(sinogram: np.ndarray, slice_idx: int, slice_axis: int = 1) -> np.ndarray:
    """Extract slice from sinogram

    Args:
        sinogram (np.ndarray): Sinogram
        slice_idx (int): ID of slice to extract
        slice_axis (int, optional): Axis along which to extract slice. Defaults to 1.

    Returns:
        np.ndarray: Slice
    """
    slice_axis = tuple([slice_idx if i == slice_axis else slice(None) for i in range(len(sinogram.shape))])
    sino_slice = np.stack(sinogram[*slice_axis])
    return sino_slice
