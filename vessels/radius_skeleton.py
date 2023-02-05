import copy

import numpy as np

from scipy import ndimage
import skimage.morphology as morphology

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
    sElement = ndimage.generate_binary_structure(binary_image.ndim, 1)
    erode_image = ndimage.morphology.binary_erosion(binary_image, sElement)
    return binary_image - erode_image


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
        list of 2 variables giving voxel size or pixel size, giving resolution in x, y

    Returns
    -------
    dict_nodes_radius : dict
        key: non-zero co-ordinate, value : radius

    Notes
    ------
    Calculates radius as distance of node on the skeleton/skeleton
    to nearest non-zero co-ordinate on the boundaries of the vessel
    """
    skeleton_image_copy = copy.deepcopy(skeleton_image)
    skeleton_image_copy[skeleton_image == 0] = 255
    skeleton_image_copy[boundary_image == 1] = 0
    eucledian_radius_image = ndimage.distance_transform_bf(skeleton_image_copy, metric='taxicab', sampling=pix_size)
    list_nzi = map(tuple, np.transpose(np.nonzero(skeleton_image)))
    return {item: eucledian_radius_image[item] for item in list_nzi}


def get_radius_slicewise(binary_vol, skeleton_vol, boundary_vol, pix_size=None, plane=0):
    """
    Returns a dictionary, image both contatining radius at a non-zero coordinate
    on centerline or skeleton
    Parameters
    ----------
    binary_vol : 3D array
        binary image of (m, n, k) shape

    skeleton_vol : 3 array
        skeletonized image of binary_image of (m, n, k) shape

    boundary_vol : 3D array
        boundaries of objects in binary_image

    pix_size : list
        list of 3 variables giving voxel size or pixel size, giving resolution in z, x, y

    plane : int
        plane = 0 look in x, y cross-section of the volume
        plane = 1 look in z, y cross-section of the volume
        plane = 2 look in z, x cross-section of the volume

    Returns
    -------
    dict_nodes_radius : dict
        key: non-zero co-ordinate, value : radius
    """
    dict_nodes_radius = {}
    for i in range(skeleton_vol.shape[plane]):
        if plane == 0:
            d = get_radius_2d(binary_vol[i, :, :], skeleton_vol[i, :, :],
                              boundary_vol[i, :, :], pix_size)
        elif plane == 1:
            d = get_radius_2d(binary_vol[:, i, :], skeleton_vol[:, i, :],
                              boundary_vol[:, i, :], pix_size)
        elif plane == 2:
            d = get_radius_2d(binary_vol[:, :, i], skeleton_vol[:, :, i],
                              boundary_vol[:, :, i], pix_size)
        new_d = {}
        for key, value in d.items():
            listed_tuple = list(key)
            listed_tuple.insert(plane, i)
            new_d[tuple(listed_tuple)] = value
        dict_nodes_radius |= new_d
    return dict_nodes_radius


def get_max_dict(list_of_dicts):
    """
    Return 2D or 3D array of vessels reconstructed
    Parameters
    ----------
    list_of_dicts : list
        list of dicts

    Returns
    -------
    max_dict : dict
        given a list of dicts having common keys,
        one dict is returned with maximum value at the common keys
    """
    return {
        key: max(list_of_dicts[i][key] for i in range(len(list_of_dicts)))
        for key, val in list_of_dicts[0].items()
    }


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
            selem = morphology.disk(int(radius)).astype(bool)
        elif len(shape) == 3:
            selem = morphology.ball(int(radius)).astype(bool)
        reconstructed_ith_image = np.zeros(shape, dtype=bool)
        reconstructed_ith_image[dest] = 1
        reconstructed_ith_image = ndimage.morphology.binary_dilation(reconstructed_ith_image, structure=selem)
        del selem
        reconstructed_image = np.logical_or(reconstructed_image, reconstructed_ith_image)
        del reconstructed_ith_image
    return reconstructed_image


def get_radius_3d(binary_vol, skeleton_vol, boundary_vol, pix_size=None):
    d = [0] * 3
    for i in range(3):
        d[i] = get_radius_slicewise(binary_vol, skeleton_vol, boundary_vol, pix_size, i)
    return get_max_dict(d)
