import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from copy import copy

from open import dicom_datasets_to_numpy, load_dicom_folder
from region_growing import RegionGrowing
from fuzzy_connectedness import FuzzyConnectedness
from profile_func import profile_func


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

    bool_palette = copy(plt.cm.get_cmap('cool'))
    bool_palette.set_bad('k', alpha=0.0)  # Color to display masked values for masked numpy arrays
    bool_palette.set_over('w', alpha=0.7)  # Color to display when value is above vmax
    bool_palette.set_under('k', alpha=0.0)  # Color to display when value is under vmin

    series_img = ax.imshow(algo.image[:, :, image_slice], interpolation='nearest', origin='bottom')
    seg_img = ax.imshow(np.zeros([algo.image.shape[0], algo.image.shape[1]], dtype=np.float32),
                        interpolation='nearest',
                        origin='bottom',
                        cmap=bool_palette,
                        norm=colors.Normalize(vmin=0, vmax=1.0))  # vmax=0.5 means True values will go over

    try:
        ax.scatter(algo.seeds[:, 0], algo.seeds[:, 1], s=2, c='r')
    except AttributeError:
        pass

    plt.colorbar(series_img, ax=ax)
    plt.colorbar(seg_img, ax=ax)
    plt.show(block=False)

    for t in algo.run():
        seg_slice = t[:, :, image_slice]
        masked_seg_img = np.ma.masked_where(seg_slice == False, seg_slice)  # Hide spels where condition is true

        seg_img.set_data(masked_seg_img)
        fig1.canvas.draw()

    plt.show(block=True)


def plot_seg(algo, image_slice=0):
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

    bool_palette = copy(plt.cm.get_cmap('hot'))
    bool_palette.set_bad('k', alpha=0.0)  # Color to display masked values for masked numpy arrays
    bool_palette.set_over('#00FF00', alpha=0.7)  # Color to display when value is above vmax
    bool_palette.set_under('#00FF00', alpha=0.0)  # Color to display when value is under vmin

    series_img = ax.imshow(algo.image[:, :, image_slice], interpolation='nearest', origin='bottom')
    seg_img = ax.imshow(np.zeros([algo.image.shape[0], algo.image.shape[1]], dtype=np.bool_),
                        interpolation='nearest',
                        origin='bottom',
                        alpha=1.0,
                        cmap=bool_palette,
                        norm=colors.Normalize(vmin=0, vmax=1.0))  # vmax=0.5 means True values will go over

    try:
        ax.scatter(algo.seeds[:, 0], algo.seeds[:, 1], s=2, c='r')
    except AttributeError:
        pass

    plt.colorbar(series_img, ax=ax)

    seg = []
    for s in algo.run():
        seg = s

    seg_slice = seg[:, :, image_slice]
    masked_seg_img = np.ma.masked_where(seg_slice == False, seg_slice)  # Hide spels where condition is true

    seg_img.set_data(masked_seg_img)
    fig1.canvas.draw()

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
    seeds = np.array([[33, 105, 0], [49, 50, 0], [157, 54, 0], [166, 127, 0]])

    algorithm = FuzzyConnectedness(series_arr, seeds, num_loops_to_yield=50)
    incremental_plot_seg(algo=algorithm, image_slice=0)


@profile_func
def main():
    np.seterr(all='raise')
    test_fuzzy_connectedness()
    test_region_growing()


if __name__ == '__main__':
    main()
