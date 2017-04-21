import numpy as np
import scipy.ndimage.filters as fi
import scipy.stats as st
import scipy.signal as sg

from segmentation_tools import check_ndimage, quick_plot
from timeit_context import timeit_context


class LevelSets(object):
    def __init__(self, image, alpha=1.0, lamb=1.0, mu=1.0, sigma=2.0, delta_t=1.0, num_loops_to_yield=100):
        self.image = image
        self.alpha = alpha
        self.lamb = lamb
        self.mu = mu
        self.sigma = sigma
        self.delta_t = delta_t
        self.num_loops_to_yield = num_loops_to_yield

    def edge_indicator1(self):
        """
        A bit slower, calculates gradient using gaussian kernel
        :return: 
        """
        filtered = fi.gaussian_gradient_magnitude(self.image, self.sigma)
        return 1.0 / (1.0 + filtered)

    def edge_indicator2(self):
        """
        A bit faster, applies gaussian filter then takes magnitude of gradient
        :return: 
        """
        filtered = fi.gaussian_filter(self.image, self.sigma)
        grad = np.gradient(filtered)
        mag = np.sqrt(grad[0]**2 + grad[1]**2 + grad[2]**2)
        return 1.0 / (1.0 + mag)

    def run(self):
        # Sanitize inputs
        try:
            check_ndimage(self.image)
        except:
            raise  # raises last exception

        g1 = self.edge_indicator1()
        g2 = self.edge_indicator2()
        quick_plot(self.image[:, :, 1])
        quick_plot(g1[:, :, 1])
        quick_plot(g2[:, :, 1])



