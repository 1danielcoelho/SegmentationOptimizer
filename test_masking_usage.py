import numpy as np
from scipy.ndimage.morphology import binary_dilation
from level_sets import magnitude_of_gradient, zero_crossing_mask
from timeit_context import timeit_context

arr = np.zeros([8, 8])
arr[:] = -2
arr[5:, 5:] = 1
arr[5, 5:] = 0
arr[5:, 5] = 0
arr[4, 4:] = -1
arr[4:, 4] = -1

mask = arr >= 0

print(arr)
print(mask)

print('gradient ')
grad1 = np.gradient(arr)
print(grad1)

print('gradient of masked')
# grad2 = [np.zeros(arr.shape), np.zeros(arr.shape)]

xx = np.zeros(arr.shape)
yy = np.zeros(arr.shape)

xx[mask] = arr[mask]
yy[mask] = arr[mask]

print(xx)

xx = np.gradient(xx, axis=1)
yy = np.gradient(yy, axis=0)
print(xx)

