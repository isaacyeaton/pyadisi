"""Calibrate cameras using a checkerboard or circle dot board. This uses
the OpenCV computer vision library to actually do the hard work.
"""

import numpy as np
import cv2
import glob

import functools
from multiprocessing import Pool


def find_objp(gridsize, dims):
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
    objp : list of an array array (npts x 3)
        Physical location of the circles or corners in a plane.
    """

    npts = dims[0] * dims[1]

    objp = np.zeros((dims[0] * dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:dims[0], 0:dims[1]].T.reshape(-1, 2)

    return npts, [objp * gridsize]


def load_images(path, func):
    """Read in images as numpy arrays and store in a list.
    """

    def _loader(fname):
        return func(cv2.imread(fname, flags=cv2.CV_LOAD_IMAGE_GRAYSCALE))

    return map(_loader, path)


def _find_circles(img, dims):
    """Find the circle grid locations from an image.

    """

    ret, circ = cv2.findCirclesGrid(img, dims, flags=cv2.CALIB_CB_SYMMETRIC_GRID)

    return ret, circ

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

    find_circles = functools.partial(_find_circles, dims=dims)

    p = Pool(nproc)
    c = p.map(find_circles, images)
    c = np.array(c)

    bad = np.where(c[:, 0] == False)[0]
    good = np.where(c[:, 0] == True)[0]
    circs = c[:, 1]  # c[good, 1]

    return circs, good, bad

def centers_array(circs):
    """Get the circe locations in a form to plot. This takes the `circs'
    output from parallel_circ_find.

    Parameters
    ----------
    circs : array of arrays

    Returns
    -------
    centers : array
        array that has the centers as a 2D array
    """

    return np.array([var.flatten().reshape(-1, 2) for var in circs])

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
        #objp = [objp for _ in range(len(imgp))]
        objp = objp * len(imgp)

    rms, intmtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp, imgp, shape[::-1], None, None)

    return rms, intmtx, dist, rvecs, tvecs


def new_int_matrix(intmxt, dist, shape):
    """Get the optimal camera matrix and region of interest.
    """

    newintmtx, roi = cv2.getOptimalNewCameraMatrix(intmxt, dist, shape, 1, shape)

    return newintmtx, roi


def save_single_cal(params):
    """
    """
    pass


def stereo_calibrate(objp, imgp1, imgp2, intmtx1, intmtx2, dist1, dist2, shape):
    """Stereo calibrate the cameras.

    Parameters
    ----------
    calibration dictionaries
    """

    if len(objp) == 1:
        objp = objp * len(imgp1)

    stereo_cal = cv2.stereoCalibrate(objectPoints=objp,
                                     imagePoints1=imgp1,
                                     imagePoints2=imgp2,
                                     cameraMatrix1=intmtx1,
                                     distCoeffs1=dist1,
                                     cameraMatrix2=intmtx2,
                                     distCoeffs2=dist2,
                                     imageSize=shape[::-1],
                                     flags=(cv2.CALIB_FIX_INTRINSIC))
    strms, int1, dis1, int2, dis2, R, T, E, F = stereo_cal
    assert(np.allclose(intmtx1, int1))
    assert(np.allclose(intmtx2, int2))
    assert(np.allclose(dist1, dis1))
    assert(np.allclose(dist2, dis2))

    return strms, R, T, E, F


def stereo_rectify(intmtx1, intmtx2, dist1, dist2, R, T, shape, alpha=-1, newsize=(0, 0)):
    """Add docs...
    """
    stereo = cv2.stereoRectify(cameraMatrix1=intmtx1, cameraMatrix2=intmtx2,
                               distCoeffs1=dist1, distCoeffs2=dist2,
                               imageSize=shape[::-1], R=R, T=T, R1=None, R2=None,
                               P1=None, P2=None, Q=None, # flags=cv2.CALIB_ZERO_DISPARITY,
                               alpha=alpha, newImageSize=newsize)
    R1, R2, P1, P2, Q, roi1, roi2 = stereo

    return R1, R2, P1, P2, Q, roi1, roi2


def dlt_coeffs(intmtx, R, T):
    """DLT coefficients given the rotation matrix and
    translation vector from stere_calibrate.

    Parameters
    ----------
    intmtx : array
        intrinsic camera matrix
    R : array
        3 x 3 rotation matrix
    T : array
        translation vector

    Returns
    -------
    L : array
        12 DLT coefficients

    Note that the DLT algorithm will use the correct 11 of them
    """

    return np.dot(intmtx, np.c_[R, T]).flatten()#[:-1]
