import numpy as np
from segmentation_tools import check_ndimage, check_seeds, get_neighbors
from timeit_context import timeit_context


class FuzzyConnectedness(object):
    def __init__(self, image, seeds, object_threshold=0.19, num_loops_to_yield=100):
        self.image = image
        self.seeds = seeds
        self.object_threshold = object_threshold
        self.num_loops_to_yield = num_loops_to_yield

    @staticmethod
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    @staticmethod
    def relative_difference(c, d):
        return abs(c - d) / (c + d)

    @staticmethod
    def average(c, d):
        return 0.5 * (c + d)

    @staticmethod
    def affinity_one(c, d, mean_ave, sigma_ave, mean_reldiff, sigma_reldiff):
        ave = FuzzyConnectedness.gaussian(FuzzyConnectedness.average(c, d),
                                          mean_ave,
                                          sigma_ave)

        rel = FuzzyConnectedness.gaussian(FuzzyConnectedness.relative_difference(c, d),
                                          mean_reldiff,
                                          sigma_reldiff)
        return min(ave, rel)

    def __call__(self, *args, **kwargs):
        # Sanitize inputs
        try:
            check_seeds(self.seeds)
            check_ndimage(self.image)
        except:
            raise  # raises last exception

        # Pad image and offset seeds
        padding_value = np.iinfo(np.int32).min + 1
        padded = np.lib.pad(self.image, 1, 'constant', constant_values=padding_value)
        self.seeds += np.array([1, 1, 1])

        # By [200, 10, 0] we mean "x=200, y=10, z=0", but if we use this to index, we'll be saying
        # "line 200, column 10, slice 0", which is the opposite of what we want, so we invert here
        for s in self.seeds:
            s[0], s[1] = s[1], s[0]

        explored = np.zeros(padded.shape, dtype=np.bool_)
        seg = np.zeros(padded.shape, dtype=np.float32)
        queue = []

        # Find out values for these
        mean_ave = 0
        sigma_ave = 0
        mean_reldiff = 0
        sigma_reldiff = 0

        # Initialize DialCache

        # Do the main loop here