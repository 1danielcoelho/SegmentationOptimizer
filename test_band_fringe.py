import numpy as np
from timeit_context import timeit_context
from more_itertools import unique_everseen
from copy import copy
from segmentation_tools import quick_plot

def get_band_indices_1d(image, band_thickness):
    return np.where(abs(image.ravel()) <= (band_thickness/2.0))[0]


def dilate_band1(indices_1d, width):
    xp = indices_1d + 1
    xn = indices_1d - 1
    yp = indices_1d + width
    yn = indices_1d - width

    full_set = set(xp) | set(xn) | set(yp) | set(yn) | set(indices_1d)

    return np.array(list(full_set))


def dilate_band2(indices_1d, width):
    xp = list(indices_1d + 1)
    xn = list(indices_1d - 1)
    yp = list(indices_1d + width)
    yn = list(indices_1d - width)

    uniques = list(unique_everseen(xp + xn + yp + yn + list(indices_1d)))

    return np.array(uniques)


def dilate_band3(indices_1d, width):
    new_list = list(indices_1d)

    # Similar time to previous methods and i'm not even removing duplicates yet
    for index in indices_1d:
        new_list.extend([index + 1, index - 1, index + width, index - width])

    return np.array(new_list)


def dilate_band4(indices_1d, width):
    num_indices = indices_1d.size

    new_arr = np.zeros(shape=[num_indices * 5], dtype=indices_1d.dtype)

    new_arr[               :1 * num_indices] = indices_1d
    new_arr[1 * num_indices:2 * num_indices] = indices_1d + 1
    new_arr[2 * num_indices:3 * num_indices] = indices_1d - 1
    new_arr[3 * num_indices:4 * num_indices] = indices_1d + width
    new_arr[4 * num_indices:5 * num_indices] = indices_1d - width

    uniques = np.unique(new_arr)

    return uniques


def dilate_band5(indices_1d, width):
    num_indices = indices_1d.size

    new_arr = np.tile(indices_1d, 5)

    new_arr[1 * num_indices:2 * num_indices] += 1
    new_arr[2 * num_indices:3 * num_indices] -= 1
    new_arr[3 * num_indices:4 * num_indices] += width
    new_arr[4 * num_indices:5 * num_indices] -= width

    uniques = np.unique(new_arr)

    return uniques


def dilate_band6(indices_1d, pattern):
    new_arr = np.tile(indices_1d, 5)
    new_arr += pattern

    uniques = np.unique(new_arr)

    return uniques


def dilate_band7(indices_1d, width):
    num_indices = indices_1d.size

    new_arr = np.tile(indices_1d, 5)

    new_arr[               :1 * num_indices] = indices_1d
    new_arr[1 * num_indices:2 * num_indices] = indices_1d + 1
    new_arr[2 * num_indices:3 * num_indices] = indices_1d - 1
    new_arr[3 * num_indices:4 * num_indices] = indices_1d + width
    new_arr[4 * num_indices:5 * num_indices] = indices_1d - width

    uniques = np.unique(new_arr)

    return uniques


#a = np.random.randint(-10, 10, size=[512, 512])

band_thickness = 10

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
        dilated_band1 = dilate_band1(indices, a.shape[1])

with timeit_context('dilate_band2'):
    for i2 in range(100):
        dilated_band2 = dilate_band2(indices, a.shape[1])

with timeit_context('dilate_band3'):
    for i3 in range(100):
        dilated_band3 = dilate_band3(indices, a.shape[1])

with timeit_context('dilate_band4'):
    for i4 in range(100):
        dilated_band4 = dilate_band4(indices, a.shape[1])

with timeit_context('dilate_band5'):
    for i5 in range(100):
        dilated_band5 = dilate_band5(indices, a.shape[1])

with timeit_context('dilate_band6'):
    num_indices = indices.size
    dilation_pattern = np.zeros(num_indices * 5, dtype=indices.dtype)
    dilation_pattern[1 * num_indices:2 * num_indices] = 1
    dilation_pattern[2 * num_indices:3 * num_indices] = - 1
    dilation_pattern[3 * num_indices:4 * num_indices] = a.shape[1]
    dilation_pattern[4 * num_indices:5 * num_indices] = - a.shape[1]

    for i6 in range(100):
        dilated_band6 = dilate_band6(indices, dilation_pattern)

with timeit_context('dilate_band7'):
    for i7 in range(100):
        dilated_band7 = dilate_band7(indices, a.shape[1])

print(dilated_band1.size)
print(dilated_band2.size)
print(dilated_band3.size)
print(dilated_band4.size)
print(dilated_band5.size)
print(dilated_band6.size)
print(dilated_band7.size)

