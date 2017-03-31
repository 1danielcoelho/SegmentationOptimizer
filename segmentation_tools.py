import numpy as np

six_neighbor_deltas = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
twenty_six_neighbor_deltas = np.array([(-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
                                       (-1, 1, -1), (-1, 1, 0), (-1, 1, 1), (0, -1, -1), (0, -1, 0), (0, -1, 1),
                                       (0, 0, -1), (0, 0, 1), (0, 1, -1), (0, 1, 0), (0, 1, 1), (1, -1, -1),
                                       (1, -1, 0), (1, -1, 1), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, -1),
                                       (1, 1, 0), (1, 1, 1)])


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
        yield point + d
