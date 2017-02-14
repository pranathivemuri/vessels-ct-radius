import copy
import os

import numpy as np

from scipy import ndimage
from scipy.misc import imread
from skimage import morphology

import kesm.analysis.skeleton.thin_volume as thin_volume
"""
Program to find radius of vessels in a 2D image, reconstruct vessels, colorcode the 2D image based on the
radius of the vessels in the image obtained using skeleton of 2D image and euclidean distance transform
WIP = Code to find maximum shortest distance in any dimension
"""


def get_boundaries_of_image(binary_image):
    """
    Return boundaries of binary image
    Parameters
    ----------
    binary_image : 2D or 3D array
        binary image of (m, n) shape to find edges/boundaries of

    Returns
    -------
    boundary_image : 2D OR 3d array
        edges of binary image, has same shape as binary_image

    Notes
    ------
    Uses a erosion based method to find edges of objects in a 2D or 3D image
    """
    assert binary_image.shape[0] != 1
    sElement = ndimage.generate_binary_structure(binary_image.ndim, 1)
    erode_image = ndimage.morphology.binary_erosion(binary_image, sElement)
    boundary_image = binary_image - erode_image
    assert np.sum(boundary_image) <= np.sum(binary_image)
    return boundary_image


def get_radius_2d(binary_image, skeleton_image, boundary_image, pix_size=None):
    """
    Returns a dictionary, image both contatining radius at a non-zero coordinate
    on centerline or skeleton
    Parameters
    ----------
    binary_image : 2D array
        binary image of (m, n) shape

    skeleton_image : 2D array
        skeletonized image of binary_image of (m, n) shape

    boundary_image : 2D array
        boundaries of objects in binary_image

    pix_size : list
        list of 2 variables giving voxel size or pixel size, giving resolution in xy, x

    Returns
    -------
    dict_nodes_radius : dict
        key: non-zero co-ordinate, value : radius

    Notes
    ------
    Calculates radius as distance of node on the skeleton/skeleton
    to nearest non-zero co-ordinate on the boundaries of the vessel
    """
    if pix_size is None:
        pix_size = [1] * skeleton_image.ndim
    skeleton_image_copy = copy.deepcopy(skeleton_image)
    skeleton_image_copy[skeleton_image == 0] = 255
    skeleton_image_copy[boundary_image == 1] = 0
    eucledian_radius_image = ndimage.distance_transform_edt(skeleton_image_copy, pix_size)
    list_nzi = map(tuple, np.transpose(np.nonzero(skeleton_image)))
    dict_nodes_radius = {item: eucledian_radius_image[item] for item in list_nzi}
    return dict_nodes_radius


def get_reconstructed_vasculature(dict_nodes_radius, shape):
    """
    Return 2D or 3D array of vessels reconstructed
    Parameters
    ----------
    dict_nodes_radius : dict
        key: non-zero co-ordinate, value : radius
    shape : tuple
        reconstructed array shape

    Returns
    -------
    reconstructed_image : 2D or 3D array
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
        reconstructed_ith_image = ndimage.morphology.binary_dilation(reconstructed_ith_image, structure=selem)
        del selem
        reconstructed_image = np.logical_or(reconstructed_image, reconstructed_ith_image)
        del reconstructed_ith_image
    return reconstructed_image


def get_normalized_values(values, normalize_upper_limit=1):
    """
    Return normalized list of elements
    Parameters
    ----------
    values : list
        list of elements to be normalized

    normalize_upper_limit : int
        upper limit to which normalize values should be scaled to, default 1
    Returns
    -------
    list
        list of normalized values scaled between 0 and normalize_upper_limit

    Notes
    -----
    Normalized as val - min_val / (max_val - min_val)
    """
    min_value = min(values)
    max_value = max(values)
    return [((value - min_value) / (max_value - min_value)) * normalize_upper_limit for value in values]


def get_radius_coded_color_vessel_skeleton(dict_nodes_radius, shape):
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
    color_coded_image = np.zeros(shape, dtype=np.float32)
    norm_radiuses = get_normalized_values(dict_nodes_radius.values(), 255)
    for dest, radius in zip(list(dict_nodes_radius.keys()), norm_radiuses):
        color_coded_image[dest] = radius
    return color_coded_image


def get_radius_slicewise(binary_vol, skeleton_vol, boundary_vol, pix_size=[1, 1], plane=0):
    """
    Returns a dictionary, image both contatining radius at a non-zero coordinate
    on centerline or skeleton
    Parameters
    ----------
    binary_image : 2D array
        binary image of (m, n) shape

    skeleton_image : 2D array
        skeletonized image of binary_image of (m, n) shape

    boundary_image : 2D array
        boundaries of objects in binary_image

    pix_size : list
        list of 2 variables giving voxel size or pixel size, giving resolution in xy, x

    Returns
    -------
    dict_nodes_radius : dict
        key: non-zero co-ordinate, value : radius
       (z, x, y) removes voxels with radius 0.0 and get radius by looking in the plane = 0 look in x, y
       plane = 1 look in z, x and plane = 2 look in z, y
    """
    dict_nodes_radius = {}
    for i in range(skeleton_vol.shape[plane]):
        if plane == 0:
            dict_nodes_radius.update(get_radius_2d(binary_vol[i, :, :], skeleton_vol[i, :, :],
                                                   boundary_vol[i, :, :], pix_size))
        elif plane == 1:
            dict_nodes_radius.update(get_radius_2d(binary_vol[:, i, :], skeleton_vol[:, i, :],
                                                   boundary_vol[:, i, :], pix_size))
        elif plane == 2:
            dict_nodes_radius.update(get_radius_2d(binary_vol[:, :, i], skeleton_vol[:, :, i],
                                                   boundary_vol[:, :, i], pix_size))
    return dict_nodes_radius


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
    dict_nodes_radius = get_radius_2d(binary_image, pix_size)
    reconstructed_image = get_reconstructed_vasculature(dict_nodes_radius, binary_image.shape)
    path_to_save = os.path.split(path)[0] + os.sep
    color_coded_image = get_radius_coded_color_vessel_skeleton(dict_nodes_radius,
                                                               path_to_save + "radius_coded_color_vessels.png",
                                                               binary_image.shape)
    path = input("enter a path to 3D binary vasculature volume")
    binary_vol = np.load(path)
    skeleton_vol = thin_volume.get_thinned(binary_vol)
    boundary_vol = get_boundaries_of_image(binary_vol)
    d = [0] * 3
    for i in range(3):
        d[i] = get_radius_slicewise(binary_vol, plane=i)
    max_radius_dict = max_radius(d)
    rv = get_reconstructed_vasculature(max_radius_dict, binary_vol.shape)
