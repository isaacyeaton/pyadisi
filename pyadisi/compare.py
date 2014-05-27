"""Compare to image sequences to get the motion of the animal.
"""

from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt

import skimage


def process_gf(img):
    return skimage.img_as_float(skimage.color.rgb2gray(img))


def subtraction(img0, img1):
    """Subtract and properly scale two images.

    Parameters
    ----------
    img0 : array
        Grayscale float image at time t
    img1 : array
        Grayscale float image at time t + 1

    Returns
    -------
    diff : array
        Grayscale float image between the two.
    """

    img0_fix = process_gf(img0)
    img1_fix = process_gf(img1)

    diff = img1_fix - img0_fix

    return diff
