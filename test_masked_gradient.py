import numpy as np
from scipy.ndimage.morphology import binary_dilation
from level_sets import magnitude_of_gradient, zero_crossing_mask
from timeit_context import timeit_context


def get_band_indices_1d(image, band_thickness):
    return np.where(abs(image.reshape(-1)) <= (band_thickness/2.0))[0]


def gradient_at_points1(image, indices_1d):
    width = image.shape[1]
    size = image.size

    # Using this instead of ravel() is more likely to produce a view instead of a copy
    raveled_image = image.reshape(-1)

    res_x = 0.5 * (raveled_image[(indices_1d + 1) % size] - raveled_image[(indices_1d - 1) % size])
    res_y = 0.5 * (raveled_image[(indices_1d + width) % size] - raveled_image[(indices_1d - width) % size])

    return [res_y, res_x]


def gradient_at_points2(image, indices_1d):
    indices_2d = np.unravel_index(indices_1d, dims=image.shape)

    # Even without doing the actual deltas this is already slower, and we'll have to check boundary conditions, etc
    res_x = 0.5 * (image[indices_2d] - image[indices_2d])
    res_y = 0.5 * (image[indices_2d] - image[indices_2d])

    return [res_y, res_x]


def gradient_at_points3(image, indices_1d):
    width = image.shape[1]

    raveled_image = image.reshape(-1)

    res_x = 0.5 * (raveled_image.take(indices_1d + 1, mode='wrap') - raveled_image.take(indices_1d - 1, mode='wrap'))
    res_y = 0.5 * (raveled_image.take(indices_1d + width, mode='wrap') - raveled_image.take(indices_1d - width, mode='wrap'))

    return [res_y, res_x]


def gradient_at_points4(image, indices_1d):
    width = image.shape[1]

    raveled_image = image.ravel()

    res_x = 0.5 * (raveled_image.take(indices_1d + 1, mode='wrap') - raveled_image.take(indices_1d - 1, mode='wrap'))
    res_y = 0.5 * (raveled_image.take(indices_1d + width, mode='wrap') - raveled_image.take(indices_1d - width, mode='wrap'))

    return [res_y, res_x]

a = np.random.randint(-10, 10, size=[512, 512])
a[:, 0] = 99
a[:, -1] = 99
a[0, :] = 99
a[-1, :] = 99

indices = get_band_indices_1d(a, 5)

# grad1 = np.gradient(a)[0]
# grad2 = gradient_at_points(image=a, indices_1d=indices)[0]
#
# grad2_res = np.zeros(shape=a.shape, dtype=np.float64)
# grad2_res[np.unravel_index(indices, grad2_res.shape)] = grad2
#
# print('grad1')
# print(grad1)
#
# print('grad2')
# print(grad2_res)


with timeit_context('full gradient'):
    for i1 in range(100):
        grad1 = np.gradient(a)

with timeit_context('gradient at points 1'):
    for i2 in range(100):
        grad2 = gradient_at_points1(image=a, indices_1d=indices)

with timeit_context('gradient at points 2'):
    for i3 in range(100):
        grad3 = gradient_at_points2(image=a, indices_1d=indices)

with timeit_context('gradient at points 3'):
    for i4 in range(100):
        grad4 = gradient_at_points3(image=a, indices_1d=indices)

with timeit_context('gradient at points 3'):
    for i5 in range(100):
        grad5 = gradient_at_points4(image=a, indices_1d=indices)

# print(a)
# print(np.gradient(a))
# print(indices)
print([grad1[0].ravel()[indices], grad1[1].ravel()[indices]])
print(grad2)
print(grad3)
print(grad4)
print(grad5)
