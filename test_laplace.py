import numpy as np
from timeit_context import timeit_context
from more_itertools import unique_everseen
from copy import copy
from segmentation_tools import quick_plot


def get_band_indices_1d(image, band_thickness):
    return np.where(abs(image.ravel()) <= (band_thickness/2.0))[0]


def laplace_at_indices1(image, indices_1d):
    pass


band_thickness = 10
# a = np.random.randint(-10, 10, size=[512, 512])
a = np.ones([512, 512]) * -10
for x in range(a.shape[0]):
    for y in range(a.shape[1]):
        if y > x + band_thickness * 0.5:
            a[x, y] = - band_thickness * 0.5
        elif y < x - band_thickness * 0.5:
            a[x, y] = band_thickness * 0.5
        else:
            a[x, y] = (x - y)

quick_plot(a)

a[:, 0] = 99
a[:, -1] = 99
a[0, :] = 99
a[-1, :] = 99

indices = get_band_indices_1d(a, band_thickness * 0.8)

with timeit_context('dilate_band1'):
    for i1 in range(100):
        dilated_band1 = laplace_at_indices1(a, indices)

print(dilated_band1)

