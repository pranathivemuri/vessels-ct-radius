import numpy as np

import copy
import operator
import time

from scipy import ndimage
from skimage import morphology

"""
   calculates radius as distance of node to nearest zero co-ordinate(edge)
   if radius is zero it is a single isolated voxel which may be
   due to noise, itself forms an edge
   these voxels can be removed through some sanity checks with
   their presence in the grey scale iage
"""


def _getBouondariesOfimage(image):
    """
       function to find boundaries/border/edges of the array/image
    """
    assert image.shape[0] != 1
    sElement = ndimage.generate_binary_structure(image.ndim, 1)
    erode_im = ndimage.morphology.binary_erosion(image, sElement)
    boundaryIm = image - erode_im
    assert np.sum(boundaryIm) <= np.sum(image)
    return boundaryIm


def getRadiusByPointsOnCenterline(inputIm, aspectRatio=None):
    """
       find radius
    """
    skeletonIm = morphology.skeletonize(inputIm)
    boundaryIm = _getBouondariesOfimage(inputIm)
    if aspectRatio is None:
        aspectRatio = [1] * skeletonIm.ndim
    skeletonImCopy = copy.deepcopy(skeletonIm)
    startt = time.time()
    skeletonImCopy[skeletonIm == 0] = 255
    skeletonImCopy[boundaryIm == 1] = 0
    distTransformedIm = ndimage.distance_transform_edt(skeletonImCopy, aspectRatio)
    listNZI = list(set(map(tuple, np.transpose(np.nonzero(skeletonIm)))))
    dictOfNodesAndRadius = {item: distTransformedIm[item] for item in listNZI}
    print("time taken to find radius at nodes is %0.3f seconds" % (time.time() - startt))
    return dictOfNodesAndRadius, distTransformedIm, skeletonIm


def getReconstructedVasculature(distTransformedIm, skeletonIm):
    startt = time.time()
    shapeConstructedIm = np.shape(distTransformedIm)
    reconstructedImage = np.zeros(shapeConstructedIm, dtype=np.uint8)
    mask = np.zeros(shapeConstructedIm, dtype=np.uint8)
    dests = map(tuple, np.transpose(np.nonzero(skeletonIm)))
    for dest in dests:
        radius = distTransformedIm[dest]
        selemDisk = morphology.disk(radius)
        mask[dest] = 1
        reconstructIthImage = ndimage.morphology.binary_dilation(skeletonIm, structure=selemDisk, iterations=-1,
                                                                 mask=mask)
        reconstructedImage = np.logical_or(reconstructedImage, reconstructIthImage)
    print("time taken to reconstruct the skeleton is %0.3f seconds" % (time.time() - startt))
    return reconstructedImage


def colorCodeByRadius(dictOfNodesAndRadius, distTransformedIm):
    channels = 3
    numDims = distTransformedIm.ndim
    colorCodedImage = np.zeros(np.shape(distTransformedIm) + (channels,), dtype=np.uint8)
    sorted_x = sorted(dictOfNodesAndRadius.items(), key=operator.itemgetter(1), reverse=True)
    for index, (key, value) in enumerate(sorted_x):
        if numDims == 3:
            x, y, z = key
            for channel in range(channels):
                colorCodedImage[z, y, x, channel] = (index + 1) * channel
        elif numDims == 2:
            x, y = key
            for channel in range(channels):
                colorCodedImage[y, x, channel] = (index + 1) * channel
    return colorCodedImage

if __name__ == '__main__':
    inputIm = np.load(input("enter a path to the CT Projection of vessels------"))
    aspectRatio = input("please enter resolution of a pixels in 2D with resolution in y and x")
    aspectRatio = [float(item) for item in aspectRatio.split(' ')]
    dictOfNodesAndRadius, distTransformedIm, skeletonIm = getRadiusByPointsOnCenterline(inputIm, aspectRatio)
    getReconstructedVasculature(distTransformedIm, skeletonIm)
    colorCodeByRadius(dictOfNodesAndRadius, distTransformedIm)
