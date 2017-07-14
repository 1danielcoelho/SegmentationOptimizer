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
from scipy.ndimage.morphology import binary_dilation

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
    """
    Alternative divergence operator restricted to 2D I found online. Calculates the divergence of a 2D vector field,
    corresponding to dFx/dx + dFy/dy
    :param nx: First dimension of the vector field, in the X direction
    :param ny: Second dimension of the vector field, in the Y direction
    :return: Single ndarray of the same shape as nx or ny, which represents a scalar field containing the divergence
    """
    _, nxx = np.gradient(nx)  # Gradient spits out [vertical, horizontal], that is, first the variation between rows
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
    phi[0] = phi[1]
    phi[-1] = phi[-2]

    if phi.ndim == 2:
        phi[:, 0] = phi[:, 1]
        phi[:, -1] = phi[:, -2]

    if phi.ndim == 3:
        phi[:, :, 0] = phi[:, :, 1]
        phi[:, :, -1] = phi[:, :, -2]


def get_or_add_default(params, param_name, default):
    if param_name in params:
        return params[param_name]
    else:
        params[param_name] = default
        return default


def zero_crossing_mask(array):
    res = np.zeros(array.shape, dtype=np.bool)

    ax0diff = np.diff(np.signbit(array), axis=0)
    res[1:] += ax0diff
    res[:-1] += ax0diff

    if res.ndim == 2:
        ax1diff = np.diff(np.signbit(array), axis=1)
        res[:, 1:] += ax1diff
        res[:, :-1] += ax1diff

    if res.ndim == 3:
        ax2diff = np.diff(np.signbit(array), axis=2)
        res[:, :, 1:] += ax2diff
        res[:, :, :-1] += ax2diff

    return res


def level_sets(image, params, phi=None, max_iter=10000, num_iter_to_update_plot=10,
               plot=False, plot_slice=0, profile=False):
    """
    Performs the Distance-Regularized Level Set Evolution (DRLSE) algorithm on 'image', using 'params' and an initial
    'phi' level set function. Will create and update a 2D plot of the algorithms' progress on slice 'plot_slice' if
    'plot' is True. Will output execution time if 'profile' is True.
    Default parameters that will be used if they are missing from 'params':
    'alpha': -5.0
    'lamb': 5.0
    'mu': 0.1
    'sigma': 0.5
    'epsilon': 1.5
    'delta_t': 1.0
    :param image: 3D ndarray containing the volume to segment using DRLSE
    :param params: Dictionary containing parameters used by the algorithm
    :param phi: 3D ndarray of the same shape as 'image' used to contain the level sets function. Should contain some
    negative value (like -2) 'inside' the desired area, and a positive value (like 2) outside. If None, will be
    initialized within the function.
    :param max_iter: Maximum number of iterations to apply to the level set function
    :param num_iter_to_update_plot: Number of iterations between plot updates. Having this too low might affect overall
    runtimes
    :param plot: When True, will cause a 2D plot to be created and updated with the algorithm's progress
    :param plot_slice: Which slice will be used to plot in case 'plot' is True
    :param profile: Whether we output the total running time of the algorithm at the end of its execution
    :return: Final 'phi' once the algorithm is complete. The zero-level contour corresponds to the final segmentation
    """

    # Sanitize inputs
    try:
        # Extract params from 'params' or get default values, and insert them into params
        alpha = get_or_add_default(params, 'alpha', -5.0)
        lamb = get_or_add_default(params, 'lamb', 5.0)
        mu = get_or_add_default(params, 'mu', 0.1)
        sigma = get_or_add_default(params, 'sigma', 0.5)
        epsilon = get_or_add_default(params, 'epsilon', 1.5)
        delta_t = get_or_add_default(params, 'delta_t', 1.0)

        check_ndimage(image)

        if phi is None:
            phi = 2 * np.ones(image.shape, dtype=np.float64)
            phi[120:150, 120:150] = -2
        else:
            check_ndimage(phi)

        if image.shape != phi.shape:
            raise AttributeError('Image and phi have different shapes!')

        if mu * delta_t >= 0.25:
            print('WARNING: mu and delta_t do not satisfy CFL condition for numerical stability (mu * delta_t < 0.25)')

        if image.ndim == 2:
            if plot_slice != 0:
                print('WARNING: image is 2D, but plot_slice is different to 0! plot_slice will be set to 0')
            plot_slice = 0
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

        plt.title('DRLSE for slice ' + str(plot_slice) + '. Iteration 0 out of ' + str(max_iter))
        plt.colorbar(series_img, ax=ax)
        plt.colorbar(seg_img, ax=ax)
        plt.show(block=False)
        plt.pause(0.001)

    # Start profiling if necessary
    start_time = 0
    if profile:
        start_time = time.time()

    # Prepare edge indicator function and its gradient (note: g itself also secretely uses gradients)
    g = edge_indicator2(image, sigma)
    g_grad = np.array(np.gradient(g))

    # Prepare narrowband mask, where True marks the narrowband spels
    # phi_mask = zero_crossing_mask(phi)
    # phi_mask = binary_dilation(phi_mask, iterations=2)

    for i in range(max_iter):
        vn_bounds(phi)

        phi_grad = np.gradient(phi)
        phi_grad_mag = magnitude_of_gradient(phi_grad)
        normalized_phi_grad = phi_grad / (phi_grad_mag + 0.0000001)

        curvature = div(normalized_phi_grad)

        dirac = delta_operator(phi, epsilon)

        # Compute stuff for distance regularization term
        a = (phi_grad_mag >= 0.0) & (phi_grad_mag <= 1.0)
        b = (phi_grad_mag > 1.0)
        ps = a * np.sin(2.0*np.pi*phi_grad_mag) / (2.0 * np.pi) + b * (phi_grad_mag - 1.0)
        dps = ((ps != 0.0) * ps + (ps == 0.0)) / ((phi_grad_mag != 0.0) * phi_grad_mag + (phi_grad_mag == 0.0))

        r_term = div(dps * phi_grad - phi_grad) + ndi.filters.laplace(phi)
        l_term = dirac * (sum(g_grad * normalized_phi_grad)) + dirac * g * curvature
        a_term = g * dirac

        phi += delta_t * (mu * r_term + lamb * l_term + alpha * a_term)

        if plot and (i+1) % num_iter_to_update_plot == 0:
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

            plt.title('DRLSE for slice ' + str(plot_slice) + '. Iteration ' + str(i+1) + ' out of ' + str(max_iter))

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

