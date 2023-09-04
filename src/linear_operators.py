from typing import List

import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import diags, kron, vstack, identity
from abc import ABC

import pyelsa as elsa

from src.filters import Filter


class IdentityOperator(LinearOperator):
    """Identity operator"""

    def __init__(self, shape, dtype="float32"):
        self.shape = shape
        self.dtype = dtype
        super().__init__(self.dtype, self.shape)

    def _matvec(self, x):
        return x

    def _adjoint(self):
        return self


class ConvolutionOperator(LinearOperator):
    """Convolution operator

    Args:
        shape (tuple): Shape of the operator
        kernel (Filter): 1D Kernel function in frequency domain
        dtype (str, optional): Data type. Defaults to "float32".
    """

    def __init__(self, shape: tuple, kernel: Filter, dtype="float32"):
        self.shape = np.flip(shape)
        self.dtype = dtype
        kernel_vectors = [kernel(s) for s in shape[:-1]]
        kernel = kernel_vectors[0]
        for i in range(1, len(kernel_vectors)):
            kernel = np.outer(kernel, kernel_vectors[i])

        self.axes = tuple(range(len(shape) - 1))
        self.kernel = np.fft.fftshift(kernel[..., np.newaxis], axes=self.axes)

        super().__init__(self.dtype, self.shape)

    def _matmat(self, x):
        fourier_x = np.fft.fftn(x, axes=self.axes)
        fourier_y = np.fft.fftshift(fourier_x, axes=-1) * self.kernel
        return np.real(np.fft.ifftn(np.fft.ifftshift(fourier_y, axes=-1), axes=self.axes))

    def apply_adjoint(self, x):
        fourier_x = np.fft.fftn(x, axes=self.axes)
        fourier_y = np.fft.fftshift(fourier_x, axes=-1) / self.kernel
        return np.real(np.fft.ifftn(np.fft.ifftshift(fourier_y, axes=-1), axes=self.axes))

    def _adjoint(self):
        class AdjointConvolutionOperator(LinearOperator):

            def __init__(self, forward: ConvolutionOperator):
                self.forward = forward
                super().__init__(self.forward.dtype, self.forward.shape)

            def _matmat(self, x):
                return self.forward.apply_adjoint(x)

            def _adjoint(self):
                return self.forward

        return AdjointConvolutionOperator(self)

    def _transpose(self):
        return self._adjoint()


class FiniteDifferences(LinearOperator, ABC):
    def __init__(self, shape, diagonal_values, offsets, dtype="float32"):
        self.ndim = len(shape)
        if self.ndim == 1:
            self.A = diags(diagonal_values, offsets=offsets, shape=(shape[0], shape[0]), format='csr')
        elif self.ndim == 2:
            y_dim = shape[0]
            x_dim = shape[1]
            ax = diags(diagonal_values, offsets=offsets, shape=(x_dim, x_dim), format='csr')
            ay = diags(diagonal_values, offsets=offsets, shape=(y_dim, y_dim), format='csr')
            self.A = vstack([kron(identity(y_dim), ax, format='csr'), kron(ay, identity(x_dim), format='csr')])
        else:
            raise ValueError("Only 1D and 2D operators are supported")
        self.shape = self.A.shape
        self.dtype = dtype
        super().__init__(self.dtype, self.shape)

    def _matvec(self, x):
        return self.A @ x

    def apply_adjoint(self, x):
        return self.A.T @ x

    def _adjoint(self):
        class AdjointFiniteDifferences(LinearOperator):
            def __init__(self, forward: FiniteDifferences):
                self.forward = forward
                super().__init__(self.forward.dtype, np.flip(self.forward.shape))

            def _matvec(self, x):
                return self.forward.apply_adjoint(x)

            def _adjoint(self):
                return self.forward

        return AdjointFiniteDifferences(self)


class ForwardDifferences(FiniteDifferences):
    """Forward differences operator of first order"""

    def __init__(self, shape, dtype="float32"):
        offsets = [0, 1]
        diagonal_values = [-1, 1]
        super().__init__(shape, diagonal_values, offsets, dtype)


class BackwardDifferences(FiniteDifferences):
    """Backward differences operator of first order"""

    def __init__(self, shape, dtype="float32"):
        offsets = [-1, 0]
        diagonal_values = [-1, 1]
        super().__init__(shape, diagonal_values, offsets, dtype)


class CentralDifferences(FiniteDifferences):
    """Central differences operator of second order"""

    def __init__(self, shape, dtype="float32"):
        offsets = [-1, 1]
        diagonal_values = [-0.5, 0.5]
        super().__init__(shape, diagonal_values, offsets, dtype)


class StackedOperator(LinearOperator):
    """Stacked operator

    Args:
        operators (list): List of operators to stack
    """
    def __init__(self, operators: List[LinearOperator]):
        self.operators = operators
        self.dtype = "float32"
        self.shape = (np.sum([op.shape[0] for op in operators], axis=0), operators[0].shape[1])
        super().__init__(self.dtype, self.shape)

    def _matvec(self, x):
        return np.hstack([op @ x for op in self.operators])

    def apply_adjoint(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros(self.shape[1])
        start = 0
        for op in self.operators:
            result += op.T @ x[start:start + op.shape[0]]
            start += op.shape[0]
        return result

    def _adjoint(self):
        class AdjointStackedOperator(LinearOperator):
            def __init__(self, forward):
                self.forward = forward
                super().__init__(self.forward.dtype, np.flip(self.forward.shape))

            def _matvec(self, x):
                return self.forward.apply_adjoint(x)

            def _adjoint(self):
                return self.forward

        return AdjointStackedOperator(self)


class XrayOperator(LinearOperator):
    """X-ray Operator

    Apply the 2- or 3-dimensional X-ray transform (and adjoint) to the signal

    By applying the operator directly to a vector (often referred to as forward
    projection), it is the simulation of the physical process of X-rays
    traversing an object. The result is a so-called sinogram. Is is the stack of
    X-ray shadows of the desired object from all given projection angles (this
    version only supports circular trajectories).

    The adjoint operation is the so-called backward projection. It takes the
    sinogram as an input, and creates an object in the image/volume domain.

    Given the`m x n` sized X-ray operator, `m` is equal to the number of
    detector pixels times the number of projection angles. `n` is the number of
    pixels/voxels in the image/volume.

    Parameters
    ----------
    vol_shape : :obj:`np.ndarray`
        Size of the image/volume
    sino_shape : :obj:`np.ndarray`
        Size of the sinogram
    thetas : :obj:`np.ndarray`
        List of projection angles in degree
    s2c : :obj:`float32`
        Distance from source to center of rotation
    c2d : :obj:`float32`
        Distance from center of rotation to detector
    vol_spacing : :obj:`np.ndarray`, optional
        Spacing of the image/volume, i.e. size of each pixel/voxel. By default,
        unit size is assumed.
    sino_spacing : :obj:`np.ndarray`, optional
        Spacing of the sinogram, i.e., size of each detector pixel. By default,
        unit size is assumed.
    cor_offset : :obj:`np.ndarray`, optional
        Offset of the center of rotation. By default, no offset is applied.
    pp_offset : :obj:`np.ndarray`, optional
        Offset of the principal point. By default, no offset is applied.
    projection_method : :obj:`str`, optional
        Projection method used for the forward and backward projections. By
        default, the interpolation/Joseph's method ('josephs') is used. Can also
        be 'siddons', for the line intersection length methods often referred
        to as Siddons method.
    dtype : :obj:`float32`, optional
        Type of elements in an input array.
    """

    def __init__(
            self,
            vol_shape,
            sino_shape,
            thetas,
            s2c,
            c2d,
            vol_spacing=None,
            sino_spacing=None,
            cor_offset=None,
            pp_offset=None,
            projection_method="josephs",
            dtype="float32",
    ):
        self.vol_shape = np.array(vol_shape)
        self.sino_shape = np.array(sino_shape)

        # Sinogram is of the same dimension as volume (i.e. it's a stack
        # of (n-1)-dimensional projection)
        if self.vol_shape.size != (self.sino_shape.size + 1):
            raise RuntimeError(
                f"Volume and sinogram must be n and (n-1) dimensional (is {self.vol_shape.size} and {self.sino_shape.size})"
            )

        self.ndim = np.size(vol_shape)

        self.thetas = np.array(thetas)

        # thetas needs to be a 1D array / list
        if self.thetas.ndim != 1:
            raise RuntimeError(
                f"angles must be a 1D array or list (is {self.thetas.ndim})"
            )

        self.s2c = s2c
        self.c2d = c2d

        self.vol_spacing = (
            np.ones(self.ndim) if vol_spacing is None else np.array(vol_spacing)
        )
        self.sino_spacing = (
            np.ones(self.ndim - 1) if sino_spacing is None else np.array(sino_spacing)
        )
        self.cor_offset = (
            np.zeros(self.ndim) if cor_offset is None else np.array(cor_offset)
        )
        self.pp_offset = (
            np.zeros(self.ndim - 1) if pp_offset is None else np.array(pp_offset)
        )

        # Some more sanity checking
        if self.vol_spacing.size != self.ndim:
            raise RuntimeError(
                f"Array containing spacing of volume is of the wrong size (is {self.vol_spacing.size}, expected {self.ndim})"
            )

        if self.cor_offset.size != self.ndim:
            raise RuntimeError(
                f"Array containing offset of center of rotation is of the wrong size (is {self.cor_offset.size}, expected {self.ndim})"
            )

        if self.sino_spacing.size != self.ndim - 1:
            raise RuntimeError(
                f"Array containing spacing of detector is of the wrong size (is {self.sino_spacing.size}, expected {self.ndim - 1})"
            )

        if self.pp_offset.size != self.ndim - 1:
            raise RuntimeError(
                f"Array containing principal point offset is of the wrong size (is {self.pp_offset.size}, expected {self.ndim - 1})"
            )

        self.vol_descriptor = elsa.VolumeDescriptor(self.vol_shape, self.vol_spacing)
        self.sino_descriptor = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
            thetas,
            self.vol_descriptor,
            self.s2c,
            self.c2d,
            self.pp_offset,
            self.cor_offset,
            self.sino_shape,
            self.sino_spacing,
        )

        if projection_method == "josephs":
            if elsa.cudaProjectorsEnabled():
                self.A = elsa.JosephsMethodCUDA(
                    self.vol_descriptor, self.sino_descriptor
                )
            else:
                self.A = elsa.JosephsMethod(self.vol_descriptor, self.sino_descriptor)
        elif projection_method == "siddons":
            if elsa.cudaProjectorsEnabled():
                self.A = elsa.SiddonsMethodCUDA(
                    self.vol_descriptor, self.sino_descriptor
                )
            else:
                self.A = elsa.SiddonsMethod(self.vol_descriptor, self.sino_descriptor)
        else:
            raise RuntimeError(f"Unknown projection method '{projection_method}'")

        M = np.prod(sino_shape) * np.size(thetas)  # number of rows
        N = np.prod(vol_shape)  # number of columns

        self.dtype = dtype
        self.shape = (M, N)

        super().__init__(self.dtype, self.shape)

    def apply(self, x):
        """Apply the forward projection to x

        Perform or simulate the forward projection given the specified parameters.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Object to forward project
        """
        # copy/move numpy array to elsa
        ex = elsa.DataContainer(np.reshape(x, self.vol_shape, order="C"), self.vol_descriptor)

        # perform forward projection
        sino = self.A.apply(ex)

        # return a numpy array
        return np.array(sino)

    def applyAdjoint(self, sino):
        """Apply the back projection to sino

        Perform or simulate the back projection given the specified parameters.

        The returned array is a 1D-vector containing the backprojection, which
        can be recreated using `backprojection.reshape(shape, order="F")`. Where
        shape the volume/image size.

        Parameters
        ----------
        sino : :obj:`np.ndarray`
            Sinogram to back project
        """
        # copy/move sinogram to elsa
        shape = np.concatenate((self.sino_shape, np.array([np.size(self.thetas)])))
        esino = elsa.DataContainer(
            np.reshape(sino, shape, order="C"), self.sino_descriptor
        )

        # perform backward projection
        bp = self.A.applyAdjoint(esino)

        # return a numpy array
        return np.array(bp) / len(self.thetas)

    def _matvec(self, x):
        """Perform the forward projection, implement the scipy.LinearOperator interface"""
        return self.apply(x).flatten("C")

    def _adjoint(self):
        """Return the adjoint, implement the scipy.LinearOperator interface"""

        class AdjointXrayOperator(LinearOperator):
            def __init__(self, forward):
                self.forward = forward
                super().__init__(self.forward.dtype, np.flip(self.forward.shape))

            def _matvec(self, sino):
                return self.forward.applyAdjoint(sino).flatten("C")

            def _adjoint(self):
                return self.forward

        return AdjointXrayOperator(self)
