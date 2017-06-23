import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from copy import copy

from open import dicom_datasets_to_numpy, load_dicom_folder
from region_growing import RegionGrowing
from fuzzy_connectedness import FuzzyConnectedness
from level_sets import LevelSets
from profile_func import profile_func

from segmentation_tools import quick_plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sampledrlse import dslre

def incremental_plot_seg(algo, image_slice=0):
    """    
    :param algo: Some object that can be called as a generator and yields a boolean image
     Should have 'image'. It if has 'seeds', they will be plotted
    :param image_slice: Zero-based index of the slice to plot in visualization
    :return: 
    """
    fig1 = plt.figure()
    plt.set_cmap(plt.gray())  # Set grayscale color palette as default

    ax = fig1.add_subplot(111)
    ax.set_aspect('equal', 'datalim')

    seg_slice = np.zeros([algo.image.shape[0], algo.image.shape[1]], dtype=np.float32)
    masked_seg_slice = np.ma.masked_where(seg_slice == 0, seg_slice)  # Hide spels where condition is true

    if algo.image.ndim == 2:
        series_pix = algo.image
    else:
        series_pix = algo.image[:, :, image_slice]

    series_img = ax.imshow(series_pix, interpolation='nearest', origin='bottom')
    seg_img = ax.imshow(masked_seg_slice,
                        interpolation='nearest',
                        origin='bottom',
                        alpha=0.7,
                        cmap='cool',
                        norm=colors.Normalize(vmin=0.0, vmax=1.0))  # vmax=0.5 means True values will go over

    try:
        ax.scatter(algo.seeds[:, 0], algo.seeds[:, 1], s=2, c='r')
    except AttributeError:
        pass

    plt.colorbar(series_img, ax=ax)
    plt.colorbar(seg_img, ax=ax)
    plt.show(block=False)

    for t in algo.run():
        if t.ndim == 3:
            seg_slice = t[:, :, image_slice]
        else:
            seg_slice = t
        masked_seg_slice = np.ma.masked_where(seg_slice == 0, seg_slice)  # Hide spels where condition is true

        seg_img.set_data(masked_seg_slice)
        fig1.canvas.draw()
        plt.pause(0.001)

    plt.show(block=True)


def incremental_plot_level_sets(algorithm, image_slice=0):
    """    
    :param algorithm: Some object that can be called as a generator and yields a boolean image
     Should have 'image'. It if has 'seeds', they will be plotted
    :param image_slice: Zero-based index of the slice to plot in visualization
    :return: 
    """

    image = algorithm.image
    fig1 = plt.figure()
    plt.set_cmap(plt.gray())  # Set grayscale color palette as default

    ax = fig1.add_subplot(111)
    ax.set_aspect('equal', 'datalim')

    seg_slice = np.zeros([image.shape[0], image.shape[1]], dtype=np.float32)

    if algorithm.image.ndim == 2:
        series_pix = algorithm.image
    else:
        series_pix = algorithm.image[:, :, image_slice]

    series_img = ax.imshow(series_pix, interpolation='nearest', origin='bottom')
    seg_img = ax.contour(seg_slice, [-2, -1, 0, 1, 2], cmap='jet', alpha=0.6)
    # seg_img = ax.imshow(seg_slice, cmap='jet', alpha=0.6, interpolation='nearest', origin='bottom')

    plt.colorbar(series_img, ax=ax)
    cb = plt.colorbar(seg_img, ax=ax)
    plt.show(block=False)

    for t in algorithm.run():
        if t.ndim == 3:
            seg_slice = t[:, :, image_slice]
        else:
            seg_slice = t

        ax.clear()
        series_img = ax.imshow(series_pix, interpolation='nearest', origin='bottom')
        seg_img = ax.contour(seg_slice, [-2, -1, 0, 1, 2], cmap='jet', alpha=0.6)
        # seg_img = ax.imshow(seg_slice, cmap='jet', alpha=0.6, interpolation='nearest', origin='bottom')
        plt.pause(0.001)
        plt.draw()

    plt.show(block=True)


def test_region_growing():
    datasets = load_dicom_folder(r"C:\Users\Daniel\Dropbox\DICOM series\ct_head_ex - Mangled")
    series_arr, _ = dicom_datasets_to_numpy(datasets)
    seeds = np.array([[33, 105, 0], [49, 50, 0], [157, 54, 0], [166, 127, 0]])

    algorithm = RegionGrowing(series_arr, seeds, sigma=7, num_loops_to_yield=100)
    incremental_plot_seg(algo=algorithm, image_slice=0)


def test_fuzzy_connectedness():
    datasets = load_dicom_folder(r"C:\Users\Daniel\Dropbox\DICOM series\ct_head_ex - Mangled")
    series_arr, _ = dicom_datasets_to_numpy(datasets)
    seeds = np.array([[33, 105, 0], [49, 50, 0], [157, 54, 0], [166, 127, 0], [60, 180, 0], [149, 185, 0]])

    algorithm = FuzzyConnectedness(series_arr, seeds, object_threshold=0.1, num_loops_to_yield=100)
    incremental_plot_seg(algo=algorithm, image_slice=0)


def test_level_sets():
    datasets = load_dicom_folder(r"C:\Users\Daniel\Dropbox\DICOM series\ct_head_ex - Mangled")
    series_arr, _ = dicom_datasets_to_numpy(datasets)

    other_arr = np.zeros([50, 50, 10])

    (width, height, depth) = other_arr.shape
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if x > 10 and x < 30 and y > 10 and y < 30 and z > 2 and z <= 5:
                    other_arr[x, y, z] = 100

    algorithm = LevelSets(other_arr)
    # algorithm.run()
    # incremental_plot_seg(algo=algorithm, image_slice=10)
    incremental_plot_level_sets(algorithm=algorithm, image_slice=3)


# @profile_func
def main():
    # np.seterr(all='raise')
    # test_fuzzy_connectedness()
    # test_region_growing()
    test_level_sets()


if __name__ == '__main__':
    main()
