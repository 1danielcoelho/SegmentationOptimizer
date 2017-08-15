import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

six_neighbor_deltas = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
twenty_six_neighbor_deltas = np.array([(-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
                                       (-1, 1, -1), (-1, 1, 0), (-1, 1, 1), (0, -1, -1), (0, -1, 0), (0, -1, 1),
                                       (0, 0, -1), (0, 0, 1), (0, 1, -1), (0, 1, 0), (0, 1, 1), (1, -1, -1),
                                       (1, -1, 0), (1, -1, 1), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, -1),
                                       (1, 1, 0), (1, 1, 1)])
padding_value = np.iinfo(np.int32).min + 1


def quick_plot(ndimage, title=""):
    fig1 = plt.figure()
    plt.set_cmap(plt.gray())  # Set grayscale color palette as default

    ax = fig1.add_subplot(111)
    ax.set_aspect('equal', 'datalim')
    img = ax.imshow(ndimage, interpolation='nearest', origin='bottom')
    plt.title(title)
    plt.colorbar(img, ax=ax)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show(block=False)
    plt.pause(0.1)


def check_seeds(seed_list):
    # None
    if seed_list is None:
        raise TypeError("seed_list is None. Needs to be a numpy array")
    # Not np.array
    if not isinstance(seed_list, np.ndarray):
        raise TypeError("seed_list not a numpy array")
    # One seed
    if seed_list.ndim == 1 and len(seed_list) == 3:
        return None
    # Multiple seeds
    elif seed_list.ndim == 2 and seed_list.shape[1] % 3 == 0:
        return None
    else:
        raise TypeError("seed_list is in an invalid shape. Needs to be (n, 3)")


def check_ndimage(image):
    # None
    if image is None:
        raise TypeError("image is None. Needs to be a numpy array")
    # Not np.array
    if not isinstance(image, np.ndarray):
        raise TypeError("image not a numpy array")
    # Unidimensional
    if image.ndim < 2:
        raise TypeError("image has less than two dimensions")


def get_neighbors(point):
    for d in six_neighbor_deltas:
        yield tuple(point + d)
