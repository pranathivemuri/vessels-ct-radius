import nose.tools
import numpy as np
from skimage import morphology, measure

import kesm.analysis.skeleton.radius_skeleton as radius_skeleton
import kesm.projects.KESMAnalysis.metrics as metrics
import kesm.analysis.phantoms.vessel_phantom as vessel_phantom
import kesm.analysis.skeleton.thin_volume as thin_volume


def _helper_radius(binary, radius, delta=1):
    if binary.ndim == 2:
        skeleton = morphology.skeletonize(binary)
    elif binary.ndim == 3:
        skeleton = thin_volume.get_thinned(binary.astype(bool))
    boundary = radius_skeleton.get_boundaries_of_image(binary)
    dict_nodes_radius = radius_skeleton.get_radius_2d(binary, skeleton, boundary)
    obtained_radius = np.mean(dict_nodes_radius.values())
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
    nose.tools.assert_equal(measure.regionprops(binary)[0].perimeter, measure.regionprops(boundary)[0].perimeter)


def test_get_radius_2d_disk():
    radius = 10
    _helper_radius(morphology.disk(radius), radius, 1)


def test_get_radius_2d_phantom():
    radius = 10
    phantom = np.amax(vessel_phantom.vessel_diagonal(radius=radius), 0)
    _helper_radius(phantom, radius, delta=1)


def test_get_radius_slicewise():
    pass


def test_get_max_dict():
    list_of_dicts = [{'a':1}, {'a': 10}, {'a':100}]
    expected = {'a': 100}
    obtained = radius_skeleton.get_max_dict(list_of_dicts)
    nose.tools.assert_dict_equal(obtained, expected)


def test_get_reconstructed_vasculature_2d_ball():
    radius = 6
    original = morphology.disk(radius)
    dict_nodes_radius = _helper_radius(original)
    predicted = radius_skeleton.get_reconstructed_vasculature(dict_nodes_radius, original.shape)
    nose.tools.assert_equal(metrics.f1_score(original, predicted), 1)


def test_get_reconstructed_vasculature_3d_disk():
    radius = 6
    original = morphology.ball(radius)
    dict_nodes_radius = _helper_radius(original)
    predicted = radius_skeleton.get_reconstructed_vasculature(dict_nodes_radius, original.shape)
    nose.tools.assert_equal(metrics.f1_score(original, predicted), 1)


def test_get_reconstructed_vasculature_2d_phantom():
    radius = 6
    original = np.amax(vessel_phantom.vessel_tree(radius), 0)
    dict_nodes_radius = _helper_radius(original)
    predicted = radius_skeleton.get_reconstructed_vasculature(dict_nodes_radius, original.shape)
    nose.tools.assert_greater_than_equal(metrics.f1_score(original, predicted), 0.7)


def test_get_reconstructed_vasculature_3d_phantom():
    radius = 6
    original = vessel_phantom.vessel_tree(radius)
    dict_nodes_radius = _helper_radius(original)
    predicted = radius_skeleton.get_reconstructed_vasculature(dict_nodes_radius, original.shape)
    nose.tools.assert_greater_than_equal(metrics.f1_score(original, predicted), 0.7)


def test_get_radius_3d_ball():
    radius = 10
    _helper_radius(morphology.ball(radius), radius, delta=1)


def test_get_radius_3d_phantom():
    radius = 10
    phantom = vessel_phantom.vessel_diagonal(radius=radius)
    _helper_radius(phantom, radius, delta=1)
