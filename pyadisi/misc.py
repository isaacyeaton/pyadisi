"""Miscellaneous functions that will probably get
moved around later.
"""

import numpy as np

def rotation_to_angles(R, deg=True):
    """Convert a rotation matrix to three angles.

    Parameters
    ----------
    R : array
        3 x 3 rotation matrix
    deg : bool, optional
        If true, return angles in degrees. Default is True

    Returns
    -------
    th : array
        Angles about x, y, and z axes

    See: https://stackoverflow.com/q/15022630/
    """

    thx = np.arctan2(R[2, 1], R[2, 2])
    thy = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    thz = np.arctan2(R[1, 0], R[0, 0])
    th = np.r_[thx, thy, thz]

    if deg:
        th = np.rad2deg(th)

    return th