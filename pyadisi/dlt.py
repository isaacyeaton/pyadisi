"""
See: https://www.mail-archive.com/floatcanvas@mithis.com/msg00513.html
Accessed Jan 20, 2014

Camera calibration and point reconstruction based on direct linear transformation (DLT).

The fundamental problem here is to find a mathematical relationship between the
 coordinates  of a 3D point and its projection onto the image plane. The DLT
 (a linear apporximation to this problem) is derived from modelling the object
 and its projection on the image plane as a pinhole camera situation.
In simplistic terms, using the pinhole camera model, it can be found by similar
 triangles the following relation between the image coordinates (u,v) and the 3D
 point (X,Y,Z):
   [ u ]   [ L1  L2  L3  L4 ] [ X ]
   [ v ] = [ L5  L6  L7  L8 ] [ Y ]
   [ 1 ]   [ L9 L10 L11 L12 ] [ Z ]
                              [ 1 ]
The matrix L is kwnown as the camera matrix or camera projection matrix. For a
 2D point (X,Y), the last column of the matrix doesn't exist. In fact, the L12
 term (or L9 for 2D DLT) is not independent from the other parameters and then
 there are only 11 (or 8 for 2D DLT) independent parameters in the DLT to be
 determined.

DLT is typically used in two steps: 1. camera calibration and 2. object (point)
 reconstruction.
The camera calibration step consists in digitizing points with known coordiantes
 in the real space.
At least 4 points are necessary for the calibration of a plane (2D DLT) and at
 least 6 points for the calibration of a volume (3D DLT). For the 2D DLT, at least
 one view of the object (points) must be entered. For the 3D DLT, at least 2
 different views of the object (points) must be entered.
These coordinates (from the object and image(s)) are inputed to the DLTcalib
 algorithm which  estimates the camera parameters (8 for 2D DLT and 11 for 3D DLT).
With these camera parameters and with the camera(s) at the same position of the
 calibration step,  we now can reconstruct the real position of any point inside
 the calibrated space (area for 2D DLT and volume for the 3D DLT) from the point
 position(s) viewed by the same fixed camera(s).

This code can perform 2D or 3D DLT with any number of views (cameras).
For 3D DLT, at least two views (cameras) are necessary.

There are more accurate (but more complex) algorithms for camera calibration that
 also consider lens distortion. For example, OpenCV and Tsai softwares have been
 ported to Python. However, DLT is classic, simple, and effective (fast) for
 most applications.

About DLT, see: http://kwon3d.com/theory/dlt/dlt.html

This code is based on different implementations and teaching material on DLT
 found in the internet.
"""

# Modified Feb 13, 2014
#Marcos Duarte - [EMAIL PROTECTED] - 04dec08

import numpy as np
from scipy import linalg


def normalization(nd, x):
    """Normalization of coordinates
    centroid to the origin and mean distance of sqrt(2 or 3)).

    Parameters
    ----------
    nd : int
        Number of dimensions (2 for 2D; 3 for 3D)
    x : array
        The data to be normalized (directions at different columns and points at rows)

    Returns
    -------
    Tr : array
        Transformation matrix (translation plus scaling)
    x : array
        Transformed data
    """

    x = np.asarray(x)
    m, s = np.mean(x,0), np.std(x)

    if nd == 2:
        Tr = np.array([[s, 0, m[0]],
                       [0, s, m[1]],
                       [0, 0, 1]])
    else:
        Tr = np.array([[s, 0, 0, m[0]],
                       [0, s, 0, m[1]],
                       [0, 0, s, m[2]],
                       [0, 0, 0, 1]])

    Tr = linalg.inv(Tr)
    x = np.dot(Tr, np.concatenate((x.T, np.ones((1, x.shape[0])))))
    x = x[0:nd, :].T

    return Tr, x


def calibrate(nd, xyz, uv):
    """Camera calibration by DLT using known object points and their image points.

    This code performs 2D or 3D DLT camera calibration with any number of views (cameras).
    For 3D DLT, at least two views (cameras) are necessary.

    Parameters
    ----------
    nd : int
        The number of dimensions of the object space: 3 for 3D DLT and 2 for 2D DLT.
    xyz : array or list
        The coordinates in the object 3D or 2D space of the calibration points.
    uv : array or list
        The coordinates in the image 2D space of these calibration points.

    The coordinates (x,y,z and u,v) are given as columns and the different points as rows.
    For the 2D DLT (object planar space), only the first 2 columns (x and y) are used.
    There must be at least 6 calibration points for the 3D DLT and 4 for the 2D DLT.

    Returns
    -------
    L : array
        The 8 or 11 parameters of the calibration matrix
    err : float
        Error of the DLT (mean residual of the DLT transformation in units of camera coordinates).
    """

    xyz = np.asarray(xyz)
    uv = np.asarray(uv)
    npts = xyz.shape[0]

    # check inputs
    if uv.shape[0] != npts:
        raise ValueError, 'xyz ({0:%d} points) and uv ({1:%d} points) have different number of points.'.format((np, uv.shape[0]))
    if (nd == 2 and xyz.shape[1] != 2) or (nd == 3 and xyz.shape[1] != 3):
        raise ValueError, 'Incorrect number of coordinates ({0:%d}) for {1:%d}D DLT (it should be {1:%d}).'.format((xyz.shape[1], nd))
    if nd == 3 and npts < 6 or nd == 2 and npts < 4:
        raise ValueError, '{0:%d}D DLT requires at least {1:%d} calibration points. Only {2:%d} points were entered.' %(nd, 2*nd, npts)

    # Normalize the data to improve the DLT quality (DLT is dependent of the system of coordinates).
    # This is relevant when there is a considerable perspective distortion.
    # Normalization: mean position at origin and mean distance equals to 1 at each direction.
    Txyz, xyzn = normalization(nd, xyz)
    Tuv, uvn = normalization(2, uv)

    A = []
    if nd == 2: # 2D DLT
        for i in range(npts):
            x, y = xyzn[i, 0], xyzn[i, 1]
            u, v = uvn[i, 0], uvn[i, 1]
            A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
            A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    elif nd == 3: # 3D DLT
        for i in range(npts):
            x, y, z = xyzn[i,0], xyzn[i,1], xyzn[i,2]
            u, v = uvn[i,0], uvn[i,1]
            A.append([x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u])
            A.append([0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v])

    A = np.asarray(A)
    U, S, Vh = linalg.svd(A)  # calculate the 11 (or 8 for 2D DLT) parameters
    L = Vh[-1, :] / Vh[-1, -1]  # parameters are in the last line of Vh and normalize them
    H = L.reshape(3, nd + 1)  # camera projection matrix

    # denormalization
    H = np.dot(np.dot(linalg.pinv(Tuv), H), Txyz)
    H = H / H[-1, -1]
    L = H.flatten(0)

    # mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates)
    uv2 = np.dot(H, np.concatenate((xyz.T, np.ones((1, xyz.shape[0])))))
    uv2 = uv2 / uv2[2, :]
    err = np.sqrt(np.mean(np.sum((uv2[0:2, :].T - uv)**2, 1)))  # mean distance

    return L, err


def reconstruct(nd, nc, Ls, uvs):
    """Reconstruction of object point from image point(s) based on the DLT parameters.

    This code performs 2D or 3D DLT point reconstruction with any number of views (cameras).
    For 3D DLT, at least two views (cameras) are necessary.

    Parameters
    ----------
    nd : int
        Number of dimensions of the object space: 3 for 3D DLT and 2 for 2D DLT.
    nc : int
        Number of cameras (views) used.
    Ls : array or list
        Camera calibration parameters of each camera. This is the calib function;
        the Ls parameters are given as columns and the rows are for different cameras.
    uvs : array
        Coordinates of the point in the image 2D space of each camera.
        The coordinates of the point are given as columns and the different views (cameras) as rows.

    Returns
    -------
    xyz : array
        point coordinates in space
    """

    Ls = np.asarray(Ls)
    uvs = np.asarray(uvs)

    # check inputs
    if Ls.ndim == 1 and nc != 1:
        raise ValueError, 'Number of views ({0:%d}) and number of sets of camera calibration parameters (1) are different.'.format(nc)
    if Ls.ndim > 1 and nc != Ls.shape[0]:
        raise ValueError, 'Number of views ({0:%d}) and number of sets of camera calibration parameters ({1:%d}) are different.'.format((nc, Ls.shape[0]))
    if nd == 3 and Ls.ndim == 1:
        raise ValueError, 'At least two sets of camera calibration parameters are needed for 3D point reconstruction.'

    if nc == 1:
        # 2D and 1 camera (view), the simplest (and fastest) case
        # One could calculate inv(H) and input that to the code to speed up things if needed.
        # (If there is only 1 camera, this transformation is all Floatcanvas2 might need)
        Hinv = linalg.inv(Ls.reshape(3, 3))
        xyz = np.dot(Hinv, [uvs[0], uvs[1], 1])  # point coordinates in space
        xyz = xyz[0:2] / xyz[2]
    else:
        M = []
        for i in range(nc):
            L = Ls[i, :]
            u, v = uvs[i, 0], uvs[i, 1]
            if nd == 2:
                M.append([L[0]-u*L[6], L[1]-u*L[7], L[2]-u*L[8]])
                M.append([L[3]-v*L[6], L[4]-v*L[7], L[5]-v*L[8]])
            elif nd == 3:
                M.append([L[0]-u*L[8], L[1]-u*L[9], L[2]-u*L[10], L[3]-u*L[11]])
                M.append([L[4]-v*L[8], L[5]-v*L[9], L[6]-v*L[10], L[7]-v*L[11]])


        U, S, Vh = np.linalg.svd(np.asarray(M))  # find the xyz coordinates
        xyz = Vh[-1, 0:-1] / Vh[-1, -1]  # point coordinates in space

    return xyz
