import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from scipy import ndimage
from scipy.misc import imread, imsave
from skimage import morphology

"""
Program to find radius of vessels in a 2D image, reconstruct vessels, colorcode the 2D image based on the
radius of the vessels in the image obtained using skeleton of 2D image and euclidean distance transform
"""


def get_boundaries_of_image(binary_image):
    """
    Return boundaries of binary image
    Parameters
    ----------
    binary_image : 2D array
        binary image of (m, n) shape to find edges/boundaries of

    Returns
    -------
    boundary_image : 2D array
        edges of binary image, has same shape as binary_image

    Notes
    ------
    Uses a erosion based method to find edges of objects in a 2D image
    """
    assert binary_image.shape[0] != 1
    sElement = ndimage.generate_binary_structure(binary_image.ndim, 1)
    erode_image = ndimage.morphology.binary_erosion(binary_image, sElement)
    boundary_image = binary_image - erode_image
    assert np.sum(boundary_image) <= np.sum(binary_image)
    return boundary_image


def get_radius_by_points_on_skeleton(binary_image, pix_size=None):
    """
    Returns an dictionary, image both contatining radius at a non-zero coordinate
    on centerline or skeleton
    Parameters
    ----------
    binary_image : 2D array
        binary image of (m, n) shape to find edges/boundaries of

    pix_size : list
        list of 3 or 2 variables giving voxel size or pixel size, giving resolution in z, y, x

    Returns
    -------
    dict_nodes_radius : dict
        key: non-zero co-ordinate, value : radius
    skeleton_image : 2D array
        skeleton/centerline of binary image of same shape as binary image
    eucledian_radius_image : 2D array
        2D float array with radius at each point on skeleton, has same shape as binary_image

    Notes
    ------
    Calculates radius as distance of node on the skeleton/skeleton
    to nearest non-zero co-ordinate on the boundaries of the vessel, all
    points on the skeleton are not included in calculating the nearest
    non-zero co-ordinate
    """
    skeleton_image = morphology.skeletonize(binary_image)
    boundary_image = get_boundaries_of_image(binary_image)
    if pix_size is None:
        pix_size = [1] * skeleton_image.ndim
    skeleton_image_copy = copy.deepcopy(skeleton_image)
    start = time.time()
    skeleton_image_copy[skeleton_image == 0] = 255
    skeleton_image_copy[boundary_image == 1] = 0
    eucledian_radius_image = ndimage.distance_transform_edt(skeleton_image_copy, pix_size)
    list_nzi = list(set(map(tuple, np.transpose(np.nonzero(skeleton_image)))))
    dict_nodes_radius = {item: eucledian_radius_image[item] for item in list_nzi}
    print("time taken to find radius at nodes is %0.3f seconds" % (time.time() - start))
    return dict_nodes_radius, eucledian_radius_image, skeleton_image


def get_reconstructed_vasculature(dict_nodes_radius, eucledian_radius_image, skeleton_image, path_to_save=None):
    """
    Return 3D volume of vessels reconstructed
    Parameters
    ----------
    dict_nodes_radius : dict
        key: non-zero co-ordinate, value : radius
    eucledian_radius_image : 2D array
        2D float array with radius at each point on skeleton, has same shape as binary_image
    skeleton_image : 2D array
        skeleton/centerline of binary image of same shape as binary image
    path_to_save: str
        absolute path to save the reconstructed 2D image at

    Returns
    -------
    networkxGraphAfter : Networkx graph
        graph with 3 vertex clique edges removed
    """
    norm_radiuses = get_normalized_values(dict_nodes_radius)
    start = time.time()
    shape_reconstructed_image = np.shape(eucledian_radius_image)
    color_coded_image = np.zeros(shape_reconstructed_image, dtype=np.float32)
    reconstructed_image = np.zeros(shape_reconstructed_image, dtype=np.uint8)
    dests = map(tuple, np.transpose(np.nonzero(skeleton_image)))
    for index, dest in enumerate(dests):
        radius = eucledian_radius_image[dest]
        selemDisk = morphology.disk(radius)
        mask = np.zeros(shape_reconstructed_image, dtype=bool)
        mask[dest] = 1
        reconstruct_ith_image = ndimage.morphology.binary_dilation(mask, structure=selemDisk)
        color_coded_image[reconstruct_ith_image != 0] = norm_radiuses[index]
        reconstructed_image = np.logical_or(reconstructed_image, reconstruct_ith_image)
    print("time taken to reconstruct the skeleton is %0.3f seconds" % (time.time() - start))
    if path_to_save is not None:
        imsave(path_to_save, reconstructed_image)
    return reconstructed_image, color_coded_image


def get_normalized_values(dict_nodes_radius):
    """
    Return 3 vertex clique removed graph
    Parameters
    ----------
    networkxGraph : Networkx graph
        graph to remove cliques from

    Returns
    -------
    networkxGraphAfter : Networkx graph
        graph with 3 vertex clique edges removed

    Notes
    ------
    Removes the longest edge in a 3 Vertex cliques,
    Special case edges are the edges with equal
    lengths that form the 3 vertex clique.
    Doesn't deal with any other cliques
    """
    radiuses = list(dict_nodes_radius.values())
    min_radius = min(radiuses)
    max_radius = max(radiuses)
    return [((radius - min_radius) / (max_radius - min_radius)) * 255 for radius in radiuses]


def get_radius_coded_color_vessel_skeleton(dict_nodes_radius, shape_image, path=None):
    """
    Return 3 vertex clique removed graph
    Parameters
    ----------
    networkxGraph : Networkx graph
        graph to remove cliques from

    Returns
    -------
    networkxGraphAfter : Networkx graph
        graph with 3 vertex clique edges removed

    Notes
    ------
    Removes the longest edge in a 3 Vertex cliques,
    Special case edges are the edges with equal
    lengths that form the 3 vertex clique.
    Doesn't deal with any other cliques
    """
    print(shape_image)
    color_coded_image = np.zeros(shape_image, dtype=np.float32)
    norm_radiuses = get_normalized_values(dict_nodes_radius)
    for dest, radius in zip(list(dict_nodes_radius.keys()), norm_radiuses):
        color_coded_image[dest] = radius
    if path is not None:
        plt.imsave(path, color_coded_image, cmap="jet")
    return color_coded_image


def plot_results(*args, **kwargs):
    """
    Return 3 vertex clique removed graph
    Parameters
    ----------
    networkxGraph : Networkx graph
        graph to remove cliques from

    Returns
    -------
    networkxGraphAfter : Networkx graph
        graph with 3 vertex clique edges removed

    Notes
    ------
    Removes the longest edge in a 3 Vertex cliques,
    Special case edges are the edges with equal
    lengths that form the 3 vertex clique.
    Doesn't deal with any other cliques
    """
    if len(args) % 2 == 0:
        fig, axes = plt.subplots(nrows=kwargs['nrows'], ncols=kwargs['ncols'])
    else:
        fig, axes = plt.subplots(nrows=1, ncols=len(args))
    for index, (ax, arg) in enumerate(zip(axes.flat, args)):
        im = ax.imshow(arg, vmin=0, vmax=1, cmap="gray")
        ax.set_title(kwargs['titles'][index])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.imsave(kwargs['path'], fig, cmap="gray")
    return fig


def compute_dice_coeffiecient(segmented_vasculature, reconstructed_vasculature, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    segmented_vasculature : 2D or 3D Boolean array
        Any array of arbitrary size.
    reconstructed_vasculature : 2D or 3D Boolean array
        Any other array of identical size as segmented_vasculature
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
    Notes
    -----
    The order of inputs is commutative.
    See https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    for more understanding
    """
    if segmented_vasculature.shape != reconstructed_vasculature.shape:
        raise ValueError("Shape mismatch: segmented_vasculature and reconstructed_vasculature must have the same shape")
    im_sum = segmented_vasculature.sum() + reconstructed_vasculature.sum()
    if im_sum == 0:
        return empty_score
    # Compute Dice coefficient
    intersection = np.logical_and(segmented_vasculature, reconstructed_vasculature)
    return 2. * intersection.sum() / im_sum


if __name__ == '__main__':
    path = input("enter a path to the CT Projection of vessels------")
    input_image = imread(path)
    pix_size = input("please enter resolution of a pixels in 2D with resolution in y and x")
    pix_size = [float(item) for item in pix_size.split(' ')]
    dict_nodes_radius, eucledian_radius_image, skeleton_image = get_radius_by_points_on_skeleton(input_image, pix_size)
    reconstructed_image = get_reconstructed_vasculature(eucledian_radius_image, skeleton_image)
    path_to_save = os.path.split(path)[0] + os.sep
    color_coded_image = get_radius_coded_color_vessel_skeleton(dict_nodes_radius,
                                                               path_to_save + "radius_coded_color_vessels.png",
                                                               skeleton_image.shape)
    plt_kwargs_dict = {'titles': ['original binary image','skeleton of original image','reconstructed image']}
    plt_kwargs_dict['path'] = path_to_save + "reconstructed_vessels.png"
    plot_results(input_image, skeleton_image, reconstructed_image, **plt_kwargs_dict)
