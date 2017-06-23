from pylab import *
from skimage import filters, io, color, morphology, exposure
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter, circle
import scipy.ndimage as ndi
from time import sleep
import os, sys
from PIL import Image

from segmentation_tools import quick_plot

def div(nx, ny):
    _, nxx = np.gradient(nx)
    nyy, _ = np.gradient(ny)
    return nxx + nyy


def delta(x, sigma):
    f = (0.5 / sigma) * (1.0 + np.cos(pi * x / sigma))
    b = (x <= sigma) & (x >= -sigma)
    return f * b


def vNBounds(phi):
    phi[0, :] = phi[1, :]
    phi[-1, :] = phi[-2, :]
    phi[:, 0] = phi[:, 1]
    phi[:, -1] = phi[:, -2]


def distReg_p1(phi, curv):
    return ndi.filters.laplace(phi) - curv


def distReg_p2(phi, dx, dy, mag):
    # dy, dx = np.gradient(phi)
    # mag = np.sqrt(dx**2+dy**2)
    a = (mag >= 0.) & (mag <= 1.)
    b = (mag > 1.)
    ps = a * np.sin(2.0 * np.pi * mag) / (2.0 * np.pi) + b * (mag - 1.0)
    dps = ((ps != 0.) * ps + (ps == 0.)) / ((mag != 0.) * mag + (mag == 0.))
    return div(dps * dx - dx, dps * dy - dy) + ndi.filters.laplace(phi)


def drlse_edge(phi, edge, lambdap, mu, alpha, epsilon, timestep, iter_inner):
    vy, vx = np.gradient(edge)
    for i2 in range(iter_inner):
        if i2 % 20 == 0:
            quick_plot(phi, 'his phi, round ' + str(i2))

        vNBounds(phi)  # edges are duplicated for no flux in or out of image
        dy, dx = np.gradient(phi)
        mag = np.sqrt((dx ** 2) + (dy ** 2))
        eps = 1e-6
        nx = dx / (mag + eps)
        ny = dy / (mag + eps)

        curv = div(nx, ny)

        # regTerm = distReg_p1(phi,curv)
        regTerm = distReg_p2(phi, dx, dy, mag)

        diracPhi = delta(phi, epsilon)

        # print nx.min(),nx.max(),curv.min(),curv.max(),regTerm.min(),regTerm.max(),diracPhi.min(),diracPhi.max()

        areaTerm = diracPhi * edge
        edgeTerm = diracPhi * (vx * nx + vy * ny) + diracPhi * edge * curv

        phi += timestep * (mu * regTerm + lambdap * edgeTerm + alpha * areaTerm)


# params
def dslre(img):
    timestep = 1.0
    mu = 0.2 / timestep
    iter_basic = 1000
    iter_refine = 10
    lambdap = 5
    alpha = -1.5  # -3
    epsilon = 1.5
    sigma = 1.5

    smoothed = filters.gaussian_filter(img, sigma)
    dy, dx = np.gradient(smoothed)
    mag = (dx ** 2) + (dy ** 2)
    edge = 1.0 / (1.0 + mag)

    c0 = 2
    initialLSF = c0 * np.ones(img.shape)
    initialLSF[15:20, 15:20] = -c0

    # initialLSF[10:55,10:75] = -c0

    # initialLSF[25:35,20:25] -= c0
    # initialLSF[25:35,40:50] -= c0
    phi = initialLSF
    drlse_edge(phi, edge, lambdap, mu, alpha, epsilon, timestep, iter_basic)
    drlse_edge(phi, edge, lambdap, mu, 0, epsilon, timestep, iter_refine)
    return phi
