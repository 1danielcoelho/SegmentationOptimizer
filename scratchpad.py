import numpy as np
from scipy.ndimage.morphology import binary_dilation

from level_sets import magnitude_of_gradient, zero_crossing_mask


arr = np.zeros([10, 10])
arr[:] = -2
arr[5:, 5:] = 1
arr[5, 5:] = 0
arr[5:, 5] = 0
arr[4, 4:] = -1
arr[4:, 4] = -1

mask = zero_crossing_mask(arr)
masked_arr = np.ma.masked_array(arr, mask=~binary_dilation(mask))

overdilated = np.ma.masked_array(arr, mask=~binary_dilation(mask, iterations=2))

print(arr)
print(masked_arr)

print('Gradient of arr')
print(np.gradient(arr)[0])

print('Gradient of masked_arr')
print(np.gradient(masked_arr)[0])

print('Gradient of masked_arr')
print(np.gradient(overdilated)[0])

# print('Magnitude of gradient of masked_arr')
# print(magnitude_of_gradient(np.gradient(masked_arr)))