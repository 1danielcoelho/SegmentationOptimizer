import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from copy import copy

from open import *
from timeit_context import timeit_context
from profile_func import profile_this
from region_growing import RegionGrowing


def incremental_plot_seg(algo):
    """    
    :param algo: Some object that can be called as a generator and yields a boolean image
     Should have 'image'. It if has 'seeds', they will be plotted
    :return: 
    """
    fig1 = plt.figure()
    plt.set_cmap(plt.gray())  # Set grayscale color palette as default

    ax = fig1.add_subplot(111)
    ax.set_aspect('equal', 'datalim')

    bool_palette = copy(plt.cm.get_cmap('gray'))
    bool_palette.set_bad('k', alpha=0.0)  # Color to display masked values for masked numpy arrays
    bool_palette.set_over('#00FF00', alpha=0.7)  # Color to display when value is above vmax
    bool_palette.set_under('k', alpha=0.0)  # Color to display when value is under vmin

    series_img = ax.imshow(algo.image[:, :, 0], interpolation='nearest', origin='bottom')
    seg_img = ax.imshow(np.zeros([algo.image.shape[0], algo.image.shape[1]], dtype=np.bool_),
                        interpolation='nearest',
                        origin='bottom',
                        alpha=1.0,
                        cmap=bool_palette,
                        norm=colors.Normalize(vmin=0, vmax=0.5))  # vmax=0.5 means True values will go over

    try:
        ax.scatter(algo.seeds[:, 0], algo.seeds[:, 1], s=2, c='r')
    except AttributeError:
        pass

    plt.colorbar(series_img, ax=ax)
    plt.show(block=False)

    for t in algo():
        seg_slice = t[:, :, 0]
        masked_seg_img = np.ma.masked_where(seg_slice == False, seg_slice)  # Hide spels where condition is true

        seg_img.set_data(masked_seg_img)
        fig1.canvas.draw()

    plt.show(block=True)


def main():
    datasets = load_dicom_folder(r"C:\Users\Daniel\Dropbox\DICOM series\ct_head_ex - Mangled")
    series_arr, _ = dicom_datasets_to_numpy(datasets)
    seeds = np.array([[33, 105, 0], [49, 50, 0], [157, 54, 0], [166, 127, 0]])

    algorithm = RegionGrowing(series_arr, seeds, sigma=7)
    incremental_plot_seg(algo=algorithm)


if __name__ == '__main__':
    main()
