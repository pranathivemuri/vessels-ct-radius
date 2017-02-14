import nose.tools

from vessels.radius_of_vessels import getRadiusByPointsOnCenterline
import kesm.analysis.phantoms.vessel_phantom as vesse_phantom


def test_getRadiusByPointsOnCenterline():
    expected_radius_list = [35, 20]
    phantom_functions = [vesse_phantom.createOneLargeVesselInclined, vesse_phantom.createOneSmallVesselInclined]
    for phantom_function, expected_radius in zip(phantom_functions, expected_radius_list):
        inputIm = phantom_function(expected_radius)
        dictRadius, _, _ = getRadiusByPointsOnCenterline(inputIm)
        obtained_avg_radius = sum(list(dictRadius.values())) / len(dictRadius)
        nose.tools.assert_almost_equal(expected_radius, obtained_avg_radius, delta=5, msg=str(phantom_function))
