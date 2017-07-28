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


def edge_indicator(i, sigma):
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


def delta_operator(x, epsilon, indices_1d):
    """
    Discrete Diract delta function. Corresponds to:
        delta(x) = 1/(2*epsilon) * (1 + cos(pi * x / epsilon)), if abs(x) <= epsilon
        delta(x) = 0, if abs(x) > epsilon
    :param x: ndarray of floats
    :param epsilon: float used for the formula
    :param indices_1d: 1d ndarray with the indices of the spels to calculate the delta operator with
    :return: 1d ndarray with the size of indices_1d containing the delta operator for the spels at those indices
    """
    selected_x = x.take(indices_1d)
    f = (0.5 / epsilon) * (1.0 + np.cos(np.pi * selected_x / epsilon))
    b = abs(selected_x) <= epsilon
    return f * b


def magnitude_of_gradient(grad_f):
    """
    Applies Pythagoras' theorem to find the magnitude of every point of the gradient
    :param grad_f: List of ndarrays, where every item of the list is one axis of the gradient
    :return: Single ndarray of the same shape as each of the items in the grad_f list
    """
    return np.sqrt(np.ufunc.reduce(np.add, [x**2 for x in grad_f]))


def div_at_indices(f, indices_1d):
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    :param f: List of ndarrays, where every item of the list is one dimension of the vector field
    :param indices_1d: 1d ndarray with the indices of the spels to calculate the divergence operator with
    :return: 1d ndarray with the size of indices_1d containing the divergence operator for the spels at those indices
    """
    xx = gradient_at_indices(f[0], indices_1d, axis=0)
    yy = gradient_at_indices(f[1], indices_1d, axis=1)
    return xx + yy


def get_band_indices_1d(image, band_thickness):
    """
    Gets the 1d (ravel'd) indices of the spels in the narrowband, assuming the image is a distance function to the
    zero level set
    :param image: n-dimensional ndarray of a distance function to a zero level set
    :param band_thickness: thickness of the entire band in 'image'
    :return: 1d ndarray with the indices of the narrowband spels of the ravel'd image
    """
    return np.where(abs(image.ravel()) <= (band_thickness/2.0))[0]


def gradient_at_indices(image, indices_1d, axis=-1):
    """
    Calculates the x,y gradients at certain spels of 'image'. The target spels are passed in a 1d array of 1d (ravel'd)
    indices, 'indices_1d'. The gradient is calculated by doing 0.5 * (image[pos + 1] - image[pos - 1]) for the x direction,
    and 0.5 * (image[pos + width] - image[pos - width]) for the y direction. Produces garbage at the edge pixels
    :param image: 2d ndarray to calculate the gradient of
    :param indices_1d: 1d ndarray with the indices of the spels of the ravel'd image to calculate the gradient of
    :param axis: axis to take the gradient in. axis=0 means the x axis (horizontal gradient), 1 means the vertical axis.
    -1 will calculate for both
    :return: 2d ndarrays, with the length of indices_1d, containing the gradient of the spels at those indices.
    Will contain np.array([grad_x]) for axis=0, np.array([grad_y]) for axis=1 or np.array([grad_y, grad_x]) for axis=-1
    """
    width = image.shape[1]
    rav_img = image.ravel()

    if axis == -1:
        res_x = 0.5 * (rav_img.take(indices_1d + 1, mode='wrap') - rav_img.take(indices_1d - 1, mode='wrap'))
        res_y = 0.5 * (rav_img.take(indices_1d + width, mode='wrap') - rav_img.take(indices_1d - width, mode='wrap'))
        return np.array([res_y, res_x])
    elif axis == 0:
        res_x = 0.5 * (rav_img.take(indices_1d + 1, mode='wrap') - rav_img.take(indices_1d - 1, mode='wrap'))
        return res_x
    elif axis == 1:
        res_y = 0.5 * (rav_img.take(indices_1d + width, mode='wrap') - rav_img.take(indices_1d - width, mode='wrap'))
        return res_y
    else:
        raise AttributeError('Invalid axis for gradient_at_points: ' + str(axis) + '. Pick 0, 1 or -1.')


def dilate_band(indices_1d, width):
    num_indices = indices_1d.size

    new_arr = np.tile(indices_1d, 5)

    new_arr[1 * num_indices:2 * num_indices] += 1
    new_arr[2 * num_indices:3 * num_indices] -= 1
    new_arr[3 * num_indices:4 * num_indices] += width
    new_arr[4 * num_indices:5 * num_indices] -= width

    uniques = np.unique(new_arr)

    return uniques


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
            c = 10

            phi = c * np.ones(image.shape, dtype=np.float64)
            middle = np.array(phi.shape) / 2.0

            phi[middle[0] - 10: middle[0] + 10, middle[1] - 10: middle[1] + 10] = -c
            phi = fi.gaussian_filter(phi, 2)

            phi[:, :2] = -c
            phi[:, -2:] = -c
            phi[:2, :] = -c
            phi[-2:, :] = -c
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

    # TODO: Ravel the image once, here. Everything past this point becomes 1D

    # TODO: Investigate how possible it is to use .take with 1d indices for 2d images

    # Prepare edge indicator function and its gradient (note: g itself also secretely uses gradients)
    g = edge_indicator(image, sigma)
    g_grad = np.gradient(g)

    # We need to keep spatial relations between spels in the narrowband. Another 'image' is the fastest way
    phi_grad_2d = [np.zeros(shape=image.shape), np.zeros(shape=image.shape)]
    phi_grad_2d_norm = [np.zeros(shape=image.shape), np.zeros(shape=image.shape)]
    dps_2d = [np.zeros(shape=image.shape), np.zeros(shape=image.shape)]

    # Initialize the band
    # Prepare narrowband mask, where True marks the narrowband spels
    zero_crossing = get_band_indices_1d(phi, 5)
    nb_indices = dilate_band(zero_crossing, image.shape[1])
    nb_indices = dilate_band(nb_indices, image.shape[1])
    nb_fringed_indices = dilate_band(nb_indices, image.shape[1])

    for i in range(max_iter):
        # Make image edges well behaved
        vn_bounds(phi)

        nb_2d_indices = np.unravel_index(nb_indices, image.shape)
        nb_fringed_2d_indices = np.unravel_index(nb_fringed_indices, image.shape)

        phi_grad_1d_yx = gradient_at_indices(phi, nb_fringed_indices)

        # TODO: Remove this once I find a better way of getting the laplacian for r_term
        phi_grad_2d[0][nb_fringed_2d_indices] = phi_grad_1d_yx[0]
        phi_grad_2d[1][nb_fringed_2d_indices] = phi_grad_1d_yx[1]

        phi_grad_1d_mag = magnitude_of_gradient(phi_grad_1d_yx)
        phi_grad_2d_norm[0][nb_fringed_2d_indices] = phi_grad_1d_yx[0] / (phi_grad_1d_mag + 0.0000001)
        phi_grad_2d_norm[1][nb_fringed_2d_indices] = phi_grad_1d_yx[1] / (phi_grad_1d_mag + 0.0000001)

        curvature_1d = div_at_indices(phi_grad_2d_norm, nb_indices)

        delta_1d = delta_operator(phi, epsilon, nb_indices)

        # Compute stuff for distance regularization term
        a = (phi_grad_1d_mag >= 0.0) & (phi_grad_1d_mag <= 1.0)
        b = (phi_grad_1d_mag > 1.0)
        ps = a * np.sin(2.0*np.pi*phi_grad_1d_mag) / (2.0 * np.pi) + b * (phi_grad_1d_mag - 1.0)
        dps = ((ps != 0.0) * ps + (ps == 0.0)) / ((phi_grad_1d_mag != 0.0) * phi_grad_1d_mag + (phi_grad_1d_mag == 0.0))

        dps_2d[0][nb_fringed_2d_indices] = dps * phi_grad_1d_yx[0] - phi_grad_1d_yx[0]
        dps_2d[1][nb_fringed_2d_indices] = dps * phi_grad_1d_yx[1] - phi_grad_1d_yx[1]

        g_nb = g.take(nb_indices)

        # Maybe this laplace filter doesn't work?
        r_term = div_at_indices(dps_2d, nb_indices) + div_at_indices(phi_grad_2d, nb_indices)
        l_term = delta_1d * \
                 (g_grad[0].take(nb_indices) * phi_grad_2d_norm[0].take(nb_indices) +
                  g_grad[1].take(nb_indices) * phi_grad_2d_norm[1].take(nb_indices) +
                  g_nb * curvature_1d)

        a_term = g_nb * delta_1d

        # quick_plot(r_term_2d, 'r_term, nb')
        # quick_plot(l_term_2d, 'l_term, nb')
        # quick_plot(a_term_2d, 'a_term, nb')

        phi[nb_2d_indices] += delta_t * (mu * r_term + lamb * l_term + alpha * a_term)

        main problem: Narrowband can't move, since only spels within the band itself are updating

        nb_indices = get_band_indices_1d(phi, 5)
        nb_indices = dilate_band(nb_indices, image.shape[1])
        nb_indices = dilate_band(nb_indices, image.shape[1])
        nb_fringed_indices = dilate_band(nb_indices, image.shape[1])

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

