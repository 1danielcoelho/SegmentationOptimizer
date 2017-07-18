import numpy as np
from timeit_context import timeit_context

phi = np.random.randint(low=-100, high=100, size=[512, 512])
# phi = np.zeros([10, 10])
# phi[:] = -2
# phi[5:, 5:] = 1
# phi[5, 5:] = 0
# phi[5:, 5] = 0
# phi[4, 4:] = -1
# phi[4:, 4] = -1

phi_mask = np.random.randint(low=0, high=2, size=phi.shape, dtype=np.bool)

with timeit_context('full array'):
    for i1 in range(1000):
        phi_masked_grad1 = np.gradient(phi)

with timeit_context('masked_array'):
    phi_masked = np.ma.masked_array(phi, ~phi_mask)
    for i3 in range(1000):
        phi_masked_grad3 = np.gradient(phi_masked)
