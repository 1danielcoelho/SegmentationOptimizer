import numpy as np
from timeit_context import timeit_context
from scipy.ndimage.morphology import binary_dilation


arr = np.random.randint(low=0, high=2, size=[512, 512, 1], dtype=np.bool)
print('Beginning')

with timeit_context('Binary dilate test'):
    for i in range(100):
        b = binary_dilation(arr, iterations=2)

# 463ms to dilate 512x512x100 ONCE

