import numpy as np
import scipy.ndimage.filters as fi
import scipy.stats as st
import scipy.signal as sg

from segmentation_tools import check_ndimage, quick_plot
from timeit_context import timeit_context


def edge_indicator1(i, sigma):
    """
    Function that highlights (with low values) sharp transitions in 'i'. Corresponds to:
        g(i) = 1 / (1 + abs(mag_gradient(gaussian_filter(i))))
    A bit slower than edge_indicator2, produces slightly smoother results
    :param i: 3d ndarray of floats
    :param sigma: float
    :return: 3d ndarray of floats of the same shape as 'i'
    """
    filtered = fi.gaussian_gradient_magnitude(i, sigma)
    return 1.0 / (1.0 + filtered**2)


def edge_indicator2(i, sigma):
    """
    Function that highlights (with low values) sharp transitions in 'i'. Corresponds to:
        g(i) = 1 / (1 + abs(mag_gradient(gaussian_filter(i)))^2)
    :param i: 3d ndarray of floats
    :param sigma: float
    :return: 3d ndarray of floats of the same shape as 'i'
    """
    filtered = fi.gaussian_filter(i, sigma)
    grad = np.gradient(filtered)
    mag_grad = magnitude_of_gradient(grad)
    return 1.0 / (1.0 + mag_grad**2)


def delta_operator(x, epsilon):
    """
    Discrete Diract delta function. Corresponds to:
        delta(x) = 1/(2*epsilon) * (1 + cos(pi * x / epsilon)), if abs(x) <= epsilon
        delta(x) = 0, if abs(x) > epsilon
    :param x: 3d ndarray of floats
    :param epsilon: float
    :return: 3d ndarray of floats of the same shape as x
    """
    const1 = 1.0 / (2.0 * epsilon)
    const2 = np.pi / epsilon
    modify_mask = abs(x) <= epsilon

    res = np.zeros(x.shape, dtype=np.float64)
    res[modify_mask] = const1 * (1 + np.cos(const2 * x[modify_mask]))
    return res


def dp(s):
    """
    Corresponds to d_p2(s) = p2'(s)/s. To avoid NANs, this function is split into three:
        dp(s) = 1, if s = 0
        dp(s) = (1/(2pi*s) * sin(2pi*s), if 0 < s < 1
        dp(s) = 1 - 1/s, if s >= 1
    :param s: 3d ndarray of floats
    :return: 3d ndarray of floats of the same shape as s
    """
    res = np.ones(s.shape, dtype=np.float64)
    less_mask = (0 < s) & (s < 1)
    larger_mask = s >= 1

    res[less_mask] = np.sin(2 * np.pi * s[less_mask]) / (2 * np.pi * s[less_mask])
    res[larger_mask] = 1.0 - 1.0 / s[larger_mask]
    return res


def magnitude_of_gradient(grad_f):
    """
    Applies Pythagoras' theorem to find the magnitude of every point of the gradient
    :param grad_f: List of ndarrays, where every item of the list is one axis of the gradient
    :return: Single ndarray of the same shape as each of the items in the grad_f list
    """
    return np.sqrt(np.ufunc.reduce(np.add, [x**2 for x in grad_f]))


def divergence(f):
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    :param f: List of ndarrays, where every item of the list is one dimension of the vector field
    :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
    """
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])


def draw_circle(array, center, radius, inside, outside):
    a, b, c = center
    mask = []

    if array.ndim == 2:
        nx, ny = array.shape
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        mask = x * x + y * y <= radius * radius

    elif array.ndim == 3:
        nx, ny, nz = array.shape
        z, y, x = np.ogrid[-a:nx - a, -b:ny - b, -c:nz - c]
        mask = x * x + y * y + z * z <= radius * radius

    array[~mask] = outside
    array[mask] = inside

    return array


class LevelSets(object):
    def __init__(self, image, alpha=1.5, lamb=5.0, mu=0.2, sigma=1.5, epsilon=1.5, delta_t=1.0, num_loops_to_yield=3):
        self.image = image
        self.phi = None

        self.alpha = alpha
        self.lamb = lamb
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon
        self.delta_t = delta_t
        self.num_loops_to_yield = num_loops_to_yield

    def run(self):
        self.image = self.image[:, :, 4]

        # Sanitize inputs
        try:
            check_ndimage(self.image)
        except:
            raise  # re-raises last exception

        self.phi = np.zeros(self.image.shape, dtype=np.float64)
        self.phi = draw_circle(self.phi, center=(100, 100, 5), radius=10, inside=2, outside=-2)
        g = edge_indicator1(self.image, self.sigma)

        max_iter = 400
        for i in range(max_iter):
            grad = np.gradient(self.phi)
            mag_grad = magnitude_of_gradient(grad)
            mag_grad = mag_grad.clip(0.0000001)  # Clip the smallest value of mag_grad to this, to avoid div/0
            delta = delta_operator(self.phi, self.epsilon)

            R = self.mu * divergence(dp(mag_grad) * grad)
            L = self.lamb * delta * divergence(g * grad/mag_grad)
            A = self.alpha * g * delta

            # quick_plot(delta, "delta")
            # quick_plot(R, "R")
            # quick_plot(L, "L")
            # quick_plot(A, "A")

            self.phi += self.delta_t * (R + L + A)

            # quick_plot(self.phi)

            if i % self.num_loops_to_yield == 0:
                yield self.phi

        yield self.phi
