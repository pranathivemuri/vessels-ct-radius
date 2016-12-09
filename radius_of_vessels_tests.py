import nose.tools

from skeleton.radius_of_vessels import getRadiusByPointsOnCenterline
import skeleton.phantom_helpers as phantom_helpers


def test_getRadiusByPointsOnCenterline():
    expected_radius_list = [35, 20]
    for expected_radius in expected_radius_list:
        inputIm = phantom_helpers.createOneLargeVesselInclined(expected_radius)
        dictRadius, _, _ = getRadiusByPointsOnCenterline(inputIm)
        obtained_avg_radius = sum(list(dictRadius.values())) / len(dictRadius)
        nose.tools.assert_almost_equal(expected_radius, obtained_avg_radius, delta=1)
