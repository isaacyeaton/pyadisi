"""GUI with pyqtgraph for digitizing images .
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
import skimage as si
from skimage import color, io, exposure

from multiprocessing import Pool

import glob
import time
import sys

# these are the suspicous imports
import pims
import h5py

pg.setConfigOptions(antialias=True)


def _quickMinMax(data):
    """We 'fix' this function so that we don't
    need a numpy array...This is only called if
    the images passed are a numpy array.
    """

    info = np.iinfo(data.dtype)
    return info.min, info.max


def pimsloader(image_paths, flip=False):
    """Load in a stack of images using pims and ducktype it
    so that we have the required methods.
    """

    def flipper(img):
        return img.swapaxes(0, 1)

    process_func = None if flip is False else flipper
    images = pims.ImageSequence(image_paths, process_func=process_func)

    # duck type on it (I feel this is a major abuse)
    nframes = len(images)
    dtype = images[0].dtype
    ndim = images[0].ndim + 1

    shape = [nframes]
    for sh in images[0].shape:
        shape.append(sh)

    size = nframes * images[0].size

    # ducktype it!
    images.dtype = dtype
    images.max = np.iinfo(dtype).max
    images.min = np.iinfo(dtype).max
    images.ndim = ndim
    images.shape = shape
    images.size = size

    return images


def hdf5loader(fname, data_path):
    """Duck type on hdf5 to get the desired attributes...I cry.
    """

    fp = h5py.File(fname)
    dat = fp[data_path]
    dat.ndim = len(dat.shape)
    dat.min = np.iinfo(dat.dtype).min
    dat.max = np.iinfo(dat.dtype).max

    return dat, fp


def imageviewer(images, crosshair=True):
    """View a stack of images (basically, a 4D numpy array).

    Parameters
    ----------
    images : thing...
        bastardized pims, hdf5, or ideally a memmaped binary file.

    Returns
    -------
    viewer : pyqgraph thing
        A neat viewer to investigate your image stack.
    """

    imv = pg.ImageView()
    imv.setWindowTitle('pyadisi is cool')

    # we are not ready to use these yet, wo we hide them :)
    imv.ui.roiBtn.hide()
    imv.ui.normBtn.hide()

    # fix quickMinMax...maybe
    if not isinstance(images, np.ndarray):
        imv.quickMinMax = _quickMinMax

    # parameters to setImage
    xvals = np.arange(images.shape[0])
    axes = {'t':0, 'x':1, 'y':2, 'c':3}

    # after it is doctored up, give it some images
    imv.setImage(images, xvals=xvals, axes=axes, autoHistogramRange=True)

    # we want to show the frame number on the image (eventually...)
    vb = imv.getView()
    label = pg.LabelItem(justify='right')
    vb.addItem(label)
    label.setText("<span style='font-size: 26pt'>frame = {0}".format(imv.currentIndex))

    # finally show (not sure when we have to do this)
    imv.show()

    # we store the data in a dictionary :)
    data = {}

    def _fill_dict(current_index, point):
        """Add values to the dictionary storing marked points.
        """

        key = '{0:05d}'.format(current_index)
        if data.has_key(key):
            data[key].append(point)
        else:
            data[key] = [point]

    # how we register mouse clicks
    def mouseClicked(evt):
        mouseclick = evt[0]
        pos = mouseclick.scenePos().toQPoint()  # https://github.com/pyqtgraph/pyqtgraph/blob/develop/pyqtgraph/Point.py#L154
        in_scene = imv.getImageItem().sceneBoundingRect().contains(pos)
        if in_scene and mouseclick.button() == 1:
            # .contains() requires a QtCore.QPointF, but we get a Point (subclassed from QtCore.QPointF) from the event
            mousePoint = vb.mapSceneToView(pos)
            print('frame {2:4d} :   (x, y): ({0:.5f}, {1:.5f})'.format(mousePoint.x(), mousePoint.y(), imv.currentIndex))
            _fill_dict(imv.currentIndex, (mousePoint.x(), mousePoint.y()))
            sys.stdout.flush()

            # push the data into the dictionary


    proxy_click = pg.SignalProxy(imv.scene.sigMouseClicked, rateLimit=60, slot=mouseClicked)


    if crosshair:
        #cross hair
        vLine = pg.InfiniteLine(angle=90, movable=False)
        hLine = pg.InfiniteLine(angle=0, movable=False)
        imv.addItem(vLine, ignoreBounds=True)
        imv.addItem(hLine, ignoreBounds=True)

        def mouseMoved(evt):
            pos = evt[0]  ## using signal proxy turns original arguments into a tuple
            if imv.getImageItem().sceneBoundingRect().contains(pos):
                mousePoint = vb.mapSceneToView(pos)
                vLine.setPos(mousePoint.x())
                hLine.setPos(mousePoint.y())

        # add the mouse callbacks for the crosshair
        proxy_chair = pg.SignalProxy(imv.scene.sigMouseMoved, rateLimit=60, slot=mouseMoved)
    else:
        proxy_chair = None

    return imv, data, proxy_chair, proxy_click