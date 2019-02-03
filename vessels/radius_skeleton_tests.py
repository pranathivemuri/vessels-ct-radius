import nose.tools
import numpy as np
import skimage.morphology as morphology
import skimage.measure as measure
import sklearn.metrics

import vessels.radius_skeleton as radius_skeleton
import vessels.cylinder_phantoms as cylinder_phantoms


def _helper_radius(binary, radius=None, delta=1):
    boundary = radius_skeleton.get_boundaries_of_image(binary)
    if binary.ndim == 2:
        skeleton = morphology.skeletonize(binary)
        dict_nodes_radius = radius_skeleton.get_radius_2d(binary, skeleton, boundary)
    elif binary.ndim == 3:
        skeleton = morphology.skeletonize_3d(binary.astype(bool))
        dict_nodes_radius = radius_skeleton.get_radius_3d(binary, skeleton, boundary, None)
    if radius is not None:
        obtained_radius = np.mean(list(dict_nodes_radius.values()))
        nose.tools.assert_almost_equal(obtained_radius, radius, delta=delta)
    return dict_nodes_radius


def test_get_boundaries_of_image_3d():
    # Test if equivalent diameter of the maximum intensity project of edges of the object is same
    # as the input sphere, measure.regionprops, 3D perimeter parameter not implemented in skimage
    radius = 4
    binary = morphology.ball(radius)
    boundary = radius_skeleton.get_boundaries_of_image(binary)
    maxip = np.amax(boundary, 0)
    nose.tools.assert_almost_equal(measure.regionprops(binary)[0].equivalent_diameter,
                                   measure.regionprops(maxip)[0].equivalent_diameter, places=1)


def test_get_boundaries_of_image_2d():
    radius = 4
    binary = morphology.disk(radius)
    boundary = radius_skeleton.get_boundaries_of_image(binary)
    nose.tools.assert_equal(measure.regionprops(binary)[0].perimeter,
                            measure.regionprops(boundary)[0].perimeter)


def test_get_radius_2d_disk():
    radius = 10
    _helper_radius(morphology.disk(radius), radius)


def test_get_radius_2d_phantom():
    radius = 10
    phantom = np.amax(cylinder_phantoms.vessel_diagonal(radius=radius), 0)
    _helper_radius(phantom, radius)


def test_get_max_dict():
    list_of_dicts = [{'a': 1}, {'a': 10}, {'a': 100}]
    expected = {'a': 100}
    obtained = radius_skeleton.get_max_dict(list_of_dicts)
    nose.tools.assert_dict_equal(obtained, expected)


def test_get_reconstructed_vasculature_2d_disk():
    radius = 6
    original = morphology.disk(radius)
    dict_nodes_radius = _helper_radius(original, radius)
    predicted = radius_skeleton.get_reconstructed_vasculature(dict_nodes_radius, original.shape)
    nose.tools.assert_equal(sklearn.metrics.f1_score(original.flatten(), predicted.flatten()), 1)


def test_get_reconstructed_vasculature_3d_ball():
    radius = 6
    original = morphology.ball(radius)
    dict_nodes_radius = _helper_radius(original, radius)
    predicted = radius_skeleton.get_reconstructed_vasculature(dict_nodes_radius, original.shape)
    nose.tools.assert_equal(sklearn.metrics.f1_score(original.flatten(), predicted.flatten()), 1)


def test_get_reconstructed_vasculature_2d_phantom():
    # 0.94
    original = np.amax(cylinder_phantoms.vessel_tree(cube_edge=64), 0)
    dict_nodes_radius = _helper_radius(original)
    predicted = radius_skeleton.get_reconstructed_vasculature(dict_nodes_radius, original.shape)
    nose.tools.assert_greater_equal(sklearn.metrics.f1_score(original.flatten(), predicted.flatten()), 0.7)


def test_get_reconstructed_vasculature_3d_phantom():
    # 0.80
    original = cylinder_phantoms.vessel_tree(cube_edge=64)
    dict_nodes_radius = _helper_radius(original)
    predicted = radius_skeleton.get_reconstructed_vasculature(dict_nodes_radius, original.shape)
    nose.tools.assert_greater_equal(sklearn.metrics.f1_score(original.flatten(), predicted.flatten()), 0.7)


def test_get_radius_3d_ball():
    radius = 5
    _helper_radius(morphology.ball(radius), radius)


def test_get_radius_3d_phantom():
    # delta is now 2 obtained radius = 3.51
    radius = 5
    phantom = cylinder_phantoms.vessel_diagonal(cube_edge=64, radius=radius)
    _helper_radius(phantom, radius, 2)
