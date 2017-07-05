import numpy as np
import scipy.ndimage.filters as fi
import scipy.stats as st
import scipy.signal as sg
import scipy.ndimage as ndi
from segmentation_tools import check_ndimage, quick_plot
from timeit_context import timeit_context
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time

#http://www.imagecomputing.org/~cmli/DRLSE/
#https://github.com/leonidk/drlse/blob/master/dlrse.py


def incremental_plot_level_sets(algorithm, image_slice=0):
    """
    :param algorithm: Some object that can be called as a generator and yields a boolean image
     Should have 'image'. It if has 'seeds', they will be plotted
    :param image_slice: Zero-based index of the slice to plot in visualization
    :return:
    """

    image = algorithm.image
    fig1 = plt.figure()
    plt.set_cmap(plt.gray())  # Set grayscale color palette as default

    ax = fig1.add_subplot(111)
    ax.set_aspect('equal', 'datalim')

    seg_slice = np.zeros([image.shape[0], image.shape[1]], dtype=np.float32)

    if algorithm.image.ndim == 2:
        series_pix = algorithm.image
    else:
        series_pix = algorithm.image[:, :, image_slice]

    series_img = ax.imshow(series_pix, interpolation='nearest', origin='bottom')
    seg_img = ax.contour(seg_slice, [-2, -1, 0, 1, 2], cmap='jet', alpha=0.6)
    # seg_img = ax.imshow(seg_slice, cmap='jet', alpha=0.6, interpolation='nearest', origin='bottom')

    plt.colorbar(series_img, ax=ax)
    cb = plt.colorbar(seg_img, ax=ax)
    plt.show(block=False)

    for t in algorithm.run():
        if t.ndim == 3:
            seg_slice = t[:, :, image_slice]
        else:
            seg_slice = t

        ax.clear()
        series_img = ax.imshow(series_pix, interpolation='nearest', origin='bottom')
        seg_img = ax.contour(seg_slice, [-2, -1, 0, 1, 2], cmap='jet', alpha=0.6)
        # seg_img = ax.imshow(seg_slice, cmap='jet', alpha=0.6, interpolation='nearest', origin='bottom')
        plt.pause(0.001)
        plt.draw()

    plt.show(block=True)


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


def vn_bounds(phi):
    phi[0, :, :] = phi[1, :, :]
    phi[-1, :, :] = phi[-2, :, :]
    phi[:, 0, :] = phi[:, 1, :]
    phi[:, -1, :] = phi[:, -2, :]
    phi[:, :, 0] = phi[:, :, 1]
    phi[:, :, -1] = phi[:, :, -2]


def level_sets(image, alpha=-5, lamb=5.0, mu=0.1, sigma=0.5, epsilon=1.5, delta_t=1.0, num_loops_to_yield=10, phi=None,
               max_iter=10000, plot=False, profile=False, plot_slice=0):

    # Sanitize inputs
    try:
        check_ndimage(image)

        if phi is None:
            phi = 2 * np.ones(image.shape, dtype=np.float64)
            phi[120:150, 120:150, 3:6] = -2
        else:
            check_ndimage(phi)

        if image.shape != phi.shape:
            raise AttributeError('Image and phi have different shapes!')

        if mu * delta_t >= 0.25:
            print('WARNING: mu and delta_t do not satisfy CFL condition for numerical stability (mu * delta_t < 0.25)')
    except:
        raise  # re-raises last exception

    # Prepare plot if necessary
    if plot:
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.set_aspect('equal', 'datalim')

        seg_slice = np.zeros([image.shape[0], image.shape[1]], dtype=np.float32)

        if image.ndim == 2:
            series_slice = image
        else:
            series_slice = image[:, :, plot_slice]

        series_img = ax.imshow(series_slice, interpolation='nearest', origin='bottom')
        seg_img = ax.contour(seg_slice, [-2, -1, 0, 1, 2], cmap='jet', alpha=0.6)
        # seg_img = ax.imshow(seg_slice, cmap='jet', alpha=0.6, interpolation='nearest', origin='bottom')

        plt.title('Distance-Regularized Level Set Evolution')
        plt.colorbar(series_img, ax=ax)
        plt.colorbar(seg_img, ax=ax)
        plt.show(block=False)

    start_time = 0
    if profile:
        start_time = time.time()

    g = edge_indicator2(image, sigma)
    [vz, vy, vx] = np.gradient(g)

    for i in range(max_iter):
        vn_bounds(phi)
        phi_z, phi_y, phi_x = np.gradient(phi)
        s = magnitude_of_gradient([phi_z, phi_y, phi_x])

        nx = phi_x / (s + 0.0000001)
        ny = phi_y / (s + 0.0000001)
        nz = phi_z / (s + 0.0000001)

        curvature = div([nz, ny, nx])

        dirac = delta_operator(phi, epsilon)

        # Compute stuff for distance regularization term
        a = (s >= 0.0) & (s <= 1.0)
        b = (s > 1.0)
        ps = a * np.sin(2.0*np.pi*s) / (2.0 * np.pi) + b * (s - 1.0)
        dps = ((ps != 0.0) * ps + (ps == 0.0)) / ((s != 0.0) * s + (s == 0.0))

        r_term = mu * (div([dps * phi_z - phi_z, dps * phi_y - phi_y, dps * phi_x - phi_x]) + ndi.filters.laplace(phi))
        l_term = lamb * (dirac * (vx * nx + vy * ny + vz * nz) + dirac * g * curvature)
        a_term = alpha * (g * dirac)

        phi += delta_t * (r_term + l_term + a_term)

        if plot and i % num_loops_to_yield == 0:
            if phi.ndim == 2:
                seg_slice = phi
                series_slice = image
            else:
                seg_slice = phi[:, :, plot_slice]
                series_slice = image[:, :, plot_slice]

            ax.clear()
            ax.imshow(series_slice, interpolation='nearest', origin='bottom')
            ax.contour(seg_slice, [-2, -1, 0, 1, 2], cmap='jet', alpha=0.6)
            # ax.imshow(seg_slice, cmap='jet', alpha=0.6, interpolation='nearest', origin='bottom')

            plt.pause(0.001)
            plt.draw()

    # Print profiling results
    if profile:
        elapsed_time = time.time() - start_time
        print('[{}] finished in {} ms'.format('Level sets', int(elapsed_time * 1000)))

    # Keep plot open when the algorithm finishes
    if plot:
        plt.show(block=True)

    return phi

