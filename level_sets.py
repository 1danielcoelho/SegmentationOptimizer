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
    return 1.0 / (1.0 + filtered)


def edge_indicator2(i, sigma):
    """
    Function that highlights (with low values) sharp transitions in 'i'. Corresponds to:
        g(i) = 1 / (1 + abs(mag_gradient(gaussian_filter(i))))
    :param i: 3d ndarray of floats
    :param sigma: float
    :return: 3d ndarray of floats of the same shape as 'i'
    """
    filtered = fi.gaussian_filter(i, sigma)
    grad = np.gradient(filtered)
    mag_grad = np.sqrt(grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2)
    return 1.0 / (1.0 + mag_grad)


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

    kill_mask = (abs(x) > epsilon)
    modify_mask = ~kill_mask

    res = np.zeros(x.shape, dtype=np.float64)
    res[kill_mask] = 0
    res[modify_mask] = const1 * (1 + np.cos(const2 * res[modify_mask]))
    return res


def dp(s):
    """
    Corresponds to d_p2(s) = p2'(s)/s. To avoid nans, this function is split into three:
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


def div(img):
    """
    Divergence operator, to be applied to a gradient
    :param img: 3 x 3d array of floats (img[0] is the x component, img[1] the y component, etc)
    :return: 3d array of floats
    """
    return img[0] + img[1] + img[2]


class LevelSets(object):
    def __init__(self, image, alpha=1.0, lamb=1.0, mu=1.0, sigma=2.0, epsilon=1.5, delta_t=1.0, num_loops_to_yield=100):
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
        # Sanitize inputs
        try:
            check_ndimage(self.image)
        except:
            raise  # raises last exception

        self.phi = np.zeros(self.image.shape, dtype=np.float64)
        g = edge_indicator2(self.image, self.sigma)

        max_iter = 100
        for i in range(max_iter):
            grad = np.gradient(self.image)
            mag_grad = np.sqrt(grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2)
            delta = delta_operator(self.phi, self.epsilon)

            quick_plot(mag_grad[:, :, 1])

            l = self.mu * div(dp(mag_grad) * grad) + \
                self.lamb * delta * div(g * grad/mag_grad) + \
                self.alpha * g * delta

            self.phi += self.delta_t * l

        quick_plot(self.phi[:, :, 1])