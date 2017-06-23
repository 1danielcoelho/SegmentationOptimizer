import numpy as np
import scipy.ndimage.filters as fi
import scipy.stats as st
import scipy.signal as sg
import scipy.ndimage as ndi

from segmentation_tools import check_ndimage, quick_plot
from timeit_context import timeit_context

#http://www.imagecomputing.org/~cmli/DRLSE/
#https://github.com/leonidk/drlse/blob/master/dlrse.py

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
    f = (0.5 / epsilon) * (1.0 + np.cos(np.pi * x / epsilon))
    b = (x <= epsilon) & (x >= -epsilon)
    return f * b


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


def div(f):
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    :param f: List of ndarrays, where every item of the list is one dimension of the vector field
    :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
    """
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])


def div2d(nx, ny):
    _, nxx = np.gradient(nx)
    nyy, _ = np.gradient(ny)
    return nxx + nyy


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


def vNBounds(phi):
    phi[0, :, :] = phi[1, :, :]
    phi[-1, :, :] = phi[-2, :, :]
    phi[:, 0, :] = phi[:, 1, :]
    phi[:, -1, :] = phi[:, -2, :]
    phi[:, :, 0] = phi[:, :, 1]
    phi[:, :, -1] = phi[:, :, -2]


class LevelSets(object):
    def __init__(self, image, alpha=-1.5, lamb=2.0, mu=0.2, sigma=0.2, epsilon=1.5, delta_t=1.0, num_loops_to_yield=3):
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
        # self.image = self.image[:, :, 2]

        # Sanitize inputs
        try:
            check_ndimage(self.image)
        except:
            raise  # re-raises last exception

        self.phi = 2 * np.ones(self.image.shape, dtype=np.float64)
        self.phi[40:45, 40:45] = -2

        g = edge_indicator2(self.image, self.sigma)

        # [vx, vy, vz] = np.gradient(g)
        [vy, vx] = np.gradient(g)

        max_iter = 500
        for i in range(max_iter):
            # if i % 20 == 0:
            #     quick_plot(self.phi, 'my phi, round ' + str(i))

            # vNBounds(self.phi)
            # phi_x, phi_y, phi_z = np.gradient(self.phi)
            phi_y, phi_x = np.gradient(self.phi)
            # s = magnitude_of_gradient([phi_x, phi_y, phi_z])
            s = magnitude_of_gradient([phi_y, phi_x])

            Nx = phi_x / (s + 0.0000001)
            Ny = phi_y / (s + 0.0000001)
            # Nz = phi_z / (s + 0.0000001)

            # curvature = div([Nx, Ny, Nz])
            curvature = div2d(Nx, Ny)

            dirac = delta_operator(self.phi, self.epsilon)

            # Compute stuff for distance regularization term
            a = (s >= 0.0) & (s <= 1.0)
            b = (s > 1.0)
            ps = a * np.sin(2.0*np.pi*s) / (2.0 * np.pi) + b * (s - 1.0)
            dps = ((ps != 0.0) * ps + (ps == 0.0)) / ((s != 0.0) * s + (s == 0.0))

            # R = self.mu * (div([dps * phi_x - phi_x, dps * phi_y - phi_y, dps * phi_z - phi_z]) +
            #                4 * ndi.filters.laplace(self.phi))

            R = self.mu * (div2d(dps * phi_x - phi_x, dps * phi_y - phi_y) +
                           ndi.filters.laplace(self.phi))

            # L = self.lamb * (dirac * (vx * Nx + vy * Ny + vz * Nz) +
            #                  dirac * g * curvature)
            L = self.lamb * (dirac * (vx * Nx + vy * Ny) +
                             dirac * g * curvature)

            A = self.alpha * (g * dirac)

            self.phi += self.delta_t * (R + L + A)

            if i % self.num_loops_to_yield == 0:
                yield self.phi

        yield self.phi
