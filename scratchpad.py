import numpy as np
from scipy.ndimage.morphology import binary_dilation

from level_sets import magnitude_of_gradient, zero_crossing_mask


phi = np.zeros([10, 10])
phi[:] = -2
phi[5:, 5:] = 1
phi[5, 5:] = 0
phi[5:, 5] = 0
phi[4, 4:] = -1
phi[4:, 4] = -1

phi_mask = zero_crossing_mask(phi)
phi_mask = binary_dilation(phi_mask, iterations=2)

# phi_masked = np.zeros(shape=phi.shape, dtype=phi.dtype)
# phi_masked[phi_mask] = phi[phi_mask]
# phi_masked_grad = np.gradient(phi_masked)

phi_masked = np.ma.masked_array(phi, ~phi_mask)
phi_masked_grad = np.gradient(phi_masked)

print(phi)

print('Gradient of arr')
print(phi_masked_grad)
