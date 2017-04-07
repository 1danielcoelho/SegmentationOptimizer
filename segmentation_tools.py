import numpy as np
from itertools import combinations

six_neighbor_deltas = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
twenty_six_neighbor_deltas = np.array([(-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
                                       (-1, 1, -1), (-1, 1, 0), (-1, 1, 1), (0, -1, -1), (0, -1, 0), (0, -1, 1),
                                       (0, 0, -1), (0, 0, 1), (0, 1, -1), (0, 1, 0), (0, 1, 1), (1, -1, -1),
                                       (1, -1, 0), (1, -1, 1), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, -1),
                                       (1, 1, 0), (1, 1, 1)])
padding_value = np.iinfo(np.int32).min + 1


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


def neighborhood_mu_sigma(aff_func, spels_pos, image):
    """
    Calculates mean and standard deviation of func applied to all unique pairs of spels
    in the neighborhood of all spels in the 'spels_pos' array
    :param aff_func: A function that computes affinity between two spels in an image
    :param spels_pos: Spels whose neighborhood will be evaluated (includes the actual spels too)
    :param image: 3D grayscale ndarray 
    :return: (mu, sigma) float tuple
    """
    # Get neighboring spel positions of all spels
    base_ground = []
    for s in spels_pos:
        for n in get_neighbors(s):
            base_ground.append(n)
        base_ground.append(s)

    # Remove repeats from neighboring positions
    # Curlies turn it into a set (no repeats)
    # Need tuples since lists are unhashable
    no_repeats = {tuple(row) for row in base_ground}

    # Convert back to ndarray for fancy indexing
    no_repeats = np.array([list(row) for row in no_repeats])

    # Evaluate affinity for all combinations
    combs = combinations(no_repeats, 2)
    combination_values = [aff_func(pair[0], pair[1], image) for pair in combs]

    mu = np.mean(combination_values)
    sigma = max(np.std(combination_values), 1)
    return mu, sigma
