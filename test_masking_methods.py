import numpy as np
from scipy.ndimage.morphology import binary_dilation
from level_sets_nb import magnitude_of_gradient, zero_crossing_mask
from timeit_context import timeit_context

arr = np.zeros([100, 100])
arr[:] = -2
arr[50:] = 2

mask = arr > 0

with timeit_context('Without masking'):
    for i in range(10000):
        b1 = arr ** 254 + 10.2
        b1 = None

    # print(b)
    # print(b.shape)

with timeit_context('With masking'):
    for i in range(10000):
        b2 = np.zeros(arr.shape)
        b2[mask] = arr[mask] ** 254 + 10.2
        b2 = None

    # print(b)
    # print(b.shape)

with timeit_context('With masked_array'):
    for i in range(10000):
        ma = np.ma.masked_array(arr, mask)
        b3 = ma ** 254 + 10.2
        b3 = None

    # print(b)
    # print(b.shape)

"""
The b produced are the same, but for some reason, we get this:

[Without masking] finished in 4042 ms
[With masking] finished in 2217 ms
[With masked_array] finished in 6562 ms

Not quite sure what happened there. Better to just use regular masking. Not we don't even need to initialize b
to the same shape and all: numpy can figure it out
"""