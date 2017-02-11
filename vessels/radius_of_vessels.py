import copy
import os
import time

# import matplotlib.pyplot as plt
import numpy as np

from scipy import ndimage
from scipy.misc import imread, imsave
from skimage import morphology

import kesm.analysis.skeleton.thin_volume as thin_volume
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


def get_reconstructed_vasculature(dict_nodes_radius, shape):
    """
    Return 3D volume of vessels reconstructed
    Parameters
    ----------
    dict_nodes_radius : dict
        key: non-zero co-ordinate, value : radius
    path_to_save: str
        absolute path to save the reconstructed 2D image at

    Returns
    -------
    reconstructed_image : 3D array
        reconstructed vasculature
    """
    reconstructed_image = np.zeros(shape, dtype=bool)
    for dest, radius in dict_nodes_radius.items():
        if len(shape) == 2:
            selem = morphology.disk(radius).astype(bool)
        elif len(shape) == 3:
            selem = morphology.ball(radius).astype(bool)
        reconstructed_ith_image = np.zeros(shape, dtype=bool)
        reconstructed_ith_image[dest] = 1
        print(selem.shape)
        reconstructed_ith_image = ndimage.morphology.binary_dilation(reconstructed_ith_image, structure=selem)
        del selem
        reconstructed_image = np.logical_or(reconstructed_image, reconstructed_ith_image)
        del reconstructed_ith_image
    return reconstructed_image


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
    color_coded_image = np.zeros(shape_image, dtype=np.float32)
    norm_radiuses = get_normalized_values(dict_nodes_radius)
    for dest, radius in zip(list(dict_nodes_radius.keys()), norm_radiuses):
        color_coded_image[dest] = radius
    # if path is not None:
        # plt.imsave(path, color_coded_image, cmap="jet")
    return color_coded_image


# def plot_results(*args, **kwargs):
#     """
#     Return 3 vertex clique removed graph
#     Parameters
#     ----------
#     networkxGraph : Networkx graph
#         graph to remove cliques from

#     Returns
#     -------
#     networkxGraphAfter : Networkx graph
#         graph with 3 vertex clique edges removed

#     Notes
#     ------
#     Removes the longest edge in a 3 Vertex cliques,
#     Special case edges are the edges with equal
#     lengths that form the 3 vertex clique.
#     Doesn't deal with any other cliques
#     """
#     if len(args) % 2 == 0:
#         fig, axes = plt.subplots(nrows=kwargs['nrows'], ncols=kwargs['ncols'])
#     else:
#         fig, axes = plt.subplots(nrows=1, ncols=len(args))
#     for index, (ax, arg) in enumerate(zip(axes.flat, args)):
#         im = ax.imshow(arg, vmin=0, vmax=1, cmap="gray")
#         ax.set_title(kwargs['titles'][index])
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#     fig.colorbar(im, cax=cbar_ax)
#     plt.imsave(kwargs['path'], fig, cmap="gray")
#     return fig


def get_radius_slicewise(binary_vol, pix_size=[1, 1], plane=0):
    """
       (z, y, x) removes voxels with radius 0.0 and get radius by looking in the plane = 0 look in x, y
       plane = 1 look in z, x and plane = 2 look in z, y
    """
    skeleton_vol = thin_volume.get_thinned(binary_vol)
    boundary_vol = get_boundaries_of_image(binary_vol)
    skeleton_vol_copy = copy.deepcopy(skeleton_vol)
    skeleton_vol_copy[skeleton_vol == 0] = 255
    skeleton_vol_copy[boundary_vol == 1] = 0
    eucledian_radius_vol = np.zeros((skeleton_vol_copy.shape))
    for i in range(skeleton_vol_copy.shape[plane]):
        if plane == 0:
            eucledian_radius_vol[i, :, :] = ndimage.distance_transform_edt(skeleton_vol_copy[i, :, :], sampling=pix_size)
        elif plane == 1:
            eucledian_radius_vol[:, i, :] = ndimage.distance_transform_edt(skeleton_vol_copy[:, i, :], sampling=pix_size)
        elif plane == 2:
            eucledian_radius_vol[:, :, i] = ndimage.distance_transform_edt(skeleton_vol_copy[:, :, i], sampling=pix_size)
    list_nzi = list(set(map(tuple, np.transpose(np.nonzero(skeleton_vol)))))
    dict_nodes_radius = list_to_dict(list_nzi, eucledian_radius_vol)
    return dict_nodes_radius, eucledian_radius_vol, skeleton_vol


def list_to_dict(list_nzi, skeletonLabelled):
    dictOfIndicesAndlabels = {item: skeletonLabelled[item] for item in list_nzi}
    return dictOfIndicesAndlabels


def max_radius(d):
    max_radius_dict = {}
    for key, val in d[0].items():
        max_radius_dict[key] = max(d[0][key], d[1][key], d[2][key])
    return max_radius_dict


if __name__ == '__main__':
    path = input("enter a path to the CT Projection of vessels------")
    binary_image = imread(path)
    pix_size = input("please enter resolution of a pixels in 2D with resolution in y and x")
    pix_size = [float(item) for item in pix_size.split(' ')]
    dict_nodes_radius, eucledian_radius_image, skeleton_image = get_radius_by_points_on_skeleton(binary_image, pix_size)
    reconstructed_image = get_reconstructed_vasculature(eucledian_radius_image, skeleton_image)
    path_to_save = os.path.split(path)[0] + os.sep
    color_coded_image = get_radius_coded_color_vessel_skeleton(dict_nodes_radius,
                                                               path_to_save + "radius_coded_color_vessels.png",
                                                               skeleton_image.shape)
    # plt_kwargs_dict = {'titles': ['original binary image','skeleton of original image','reconstructed image']}
    # plt_kwargs_dict['path'] = path_to_save + "reconstructed_vessels.png"
    # plot_results(binary_image, skeleton_image, reconstructed_image, **plt_kwargs_dict)
    path = input("enter a path to 3D binary vasculature volume")
    binary_vol = np.load(path)
    d = [0] * 3
    for i in range(3):
        d[i], _, _ = get_radius_slicewise(binary_vol, plane=i)
    max_radius_dict = max_radius(d)
    rv = get_reconstructed_vasculature(max_radius_dict, binary_vol.shape)
