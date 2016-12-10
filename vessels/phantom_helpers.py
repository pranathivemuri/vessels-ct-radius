import numpy as np
import cv2


def _create_vessels_2D(cubeEdge, p1, p2, r, noise=False):
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
    """
    given two tuples with the start and end points, create a cylinder between these points with radius r
    line goes through point (x0, y0, z0) and in direction of unit vector (u1, u2, u3)
    point on line closest to (x, y, z) is
    (X0,Y0,Z0) + ((X-X0)u1+(Y-Y0)u2+(Z-Z0)u3) (u1,u2,u3)

    (x-x0)^2 + (y-y0)^2 = r^2
    """
    stack = np.ones((cubeEdge, cubeEdge, cubeEdge)) * 20
    if p1[0] > p2[0]:  # points need to have the first point have a lower Z
        p1, p2 = p2, p1  # swap
    z = np.arange(p1[0], p2[0])
    n = len(z)
    x, y = np.linspace(p1[1], p2[1], n, dtype=int), np.linspace(p1[2], p2[2], n, dtype=int)
    color = 255  # white
    thickness = -1  # filled circle
    for i, zz in enumerate(z):
        stack[zz, :, :] = cv2.circle(stack[zz, :, :].copy(), (x[i], y[i]), r, color, thickness)
    stack = stack.astype(np.uint8)
    if noise:
        stack = _addNoise(stack)
    maxip = np.amax(stack, 0)
    maxip[maxip <= 20] = 0
    return maxip.astype(bool)
    return stack


def _addNoise(stack, level=10, sigma=3):
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
    """
    add noise defined by level (+- value, within 255) and sigma (std of gaussian kernel)
    """
    assert sigma % 2 == 1, "only odd kernel sizes are allowed"
    noise = np.random.random_integers(-level, level, stack.shape).astype(np.float64)
    for i in range(noise.shape[2]):
        noise[i, :, :] = cv2.GaussianBlur(noise[i, :, :].copy(), (sigma, sigma), 0)
    stack += noise
    # add another level of speckle, just because it's actually this hard
    noise2 = np.random.random_integers(-level, level, stack.shape).astype(np.float64)
    stack += noise2
    stack = stack.clip(0, 255)
    return stack


def createOneLargeVesselInclined(radius=35, noise=False):
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
    cubeEdge = 512
    p1, p2 = (30, 15, 0), (480, 400, cubeEdge - 30)
    maxip = _create_vessels_2D(cubeEdge, p1, p2, radius, noise)
    return maxip


def createOneSmallVesselInclined(radius=20, noise=False):
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
    cubeEdge = 512
    p1, p2 = (154, 138, 160), (94, 380, 400)
    maxip = _create_vessels_2D(cubeEdge, p1, p2, radius, noise)
    return maxip
