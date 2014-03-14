"""Calibrate cameras using a checkerboard or circle dot board. This uses
the OpenCV computer vision library to actually do the hard work.
"""

import numpy as np
import cv2
import glob

import functools
from multiprocessing import Pool


def find_objp(girdsize, dims):
    """Make an array of the object points for the grid.

    Parameters
    ----------
    gridsize : float
        Distance between circles or length of checkerboard corners
        in whatever physical units you are using.
    dims : tuple
        (width, height) of the circle grid pattern or the inner
        dimension of the checkerboard.

    Returns
    -------
    npts : int
        Number of grid calibration points
    objp : array (npts x 3)
        Physical location of the circles or corners in a plane.
    """

    npts = width * height

    objp = np.zeros((dims[0] * dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:dims[0], 0:dims[1]].T.reshape(-1, 2)

    return npts, objp * gridsize


def _find_circles(img, dims):
    """Find the circle grid locations from an image.

    """

    ret, circ = cv2.findCirclesGrid(img, dims, flags=cv2.CALIB_CB_SYMMETRIC_GRID)

    return ret, circ


def load_images(path, func):
    """Read in images as numpy arrays and store in a list.
    """

    def _loader(fname):
        return func(cv2.imread(fname, flags=cv2.CV_LOAD_IMAGE_GRAYSCALE))

    return map(_loader, path)


def parallel_circ_find(images, dims, nproc=4):
    """Use multiprocessing module to find circle pattern
    in parallel.

    Parameters
    ----------
    images : list or numpy arrays
        Preloaded images (see load_images)
    dims : tuple
        Circle dimensions

    Returns
    -------
    circs : list
        Location of the circle centers (includes None if not found)
    good : list
        Index of where valid images were found
    bad : list
        Index of images where grid could not be found
    """

    find_circles = functools.partial(find_circ, dims=dims)

    p = Pool(nproc)
    c = p.map(find_circles, images)
    c = np.array(c)

    bad = np.where(c[:, 0] == False)
    good = np.where(c[:, 0] == True)
    circs = c[:, 1]  # c[good, 1]

    return circs, good, bad


def valid_stereo_pairs(good1, good2, bad1, bad2):
    """Select indices and points for the stereo calibration. These
    images had a successful grid find in both frames.
    """

    bad = np.intersect1d(bad1, bad2)
    good = np.intersect1d(good1, good2)

    return good, bad


def single_calibrate(objp, imgp, shape):
    """Do the intrinisic calibration.
    """

    # get the correct number of object points
    if len(objp) == 1:
        objp = [objp for _ in range(len(imgp))]

    rms, intmtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp, imgp, shape[::-1], None, None)

    return rms, intmtx, dist, rvecs, tvecs

