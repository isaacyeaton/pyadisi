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


def mrawloader(fname, dtype=np.uint16, h=1024, w=1024, flip=True):
    """Load in an mraw as a memmaped numpy array.

    Parameters
    ----------
    fname : str
        Location of the binary file.
    dtype : np.dtype, default=np.uint16
        Type of stored binary data.
    h : int, default=1024
        Height of the images.
    w : int, default=1024
        Width of the images.
    flip : bool, default=True
        Whether to flip the image or not.

    Returns
    -------
    images : np.ndarray (numpy.core.memmap.memmap)
        Numpy memmaped array

    Notes
    -----
    This assumes that the images are not color. The data about the image
    dimensions and number of frames can be extracted from the .cih file
    (see pyadisi.metadata.photron).

    Examples
    --------
    >>> images = pyadisi.gui.mrawloader('~/Desktop/2014-05-29_000001.mraw')
    """

    images = np.memmap(fname, dtype, 'c').reshape(-1, h, w).swapaxes(1, 2)
    return images


def pimsloader(image_paths, flip=False, process_func=None):
    """Load in a stack of images using pims and ducktype it
    so that we have the required methods.

    Parameters
    ----------
    image_paths : str
        Path to the images to load.
    flip : bool, default=False
        Whether to flip the image or not.

    Returns
    -------
    images : pims.image_sequence.ImageSequence
        Images, but with required methods for the gui.

    Example
    -------
    >>> images = pyadisi.gui.pimsload('~/Desktop/2014-05-29_000001/*/*.tif')
    """

    def flipper(img):
        return img[::-1, :]  # .swapaxes(0, 1)

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

    Parameters
    ----------
    fname : str
        The hdf5 file to load.
    data_path : str
        The local path inside the hdf5 file to the data.

    Returns
    -------
    images : h5py._hl.dataset.Dataset
        Images with the required methods for the gui.
    fp : h5py._hl.files.File
        Open hdf5 file (so fp.close() can be used to gracefully close it).

    Notes
    -----
    This assumes the data is (time, y, x, c).

    Example
    -------
    >>> images = pyadisi.gui.pimsload('~/Desktop/2014-05-29_000001.hdf5', 'raw')
    """

    fp = h5py.File(fname)
    dat = fp[data_path]
    dat.ndim = len(dat.shape)
    dat.min = np.iinfo(dat.dtype).min
    dat.max = np.iinfo(dat.dtype).max

    return dat, fp


def imageviewer(images, crosshair=True, xvals=None):
    """View a stack of images (basically, a 4D numpy array).

    Parameters
    ----------
    images : image stack
        Bastardized pims, hdf5, or ideally a memmaped binary file.
    crosshair : bool, default=True
        Whether to show the crosshair on the images.
    xvals : np.ndarray, default=None
        The time or frame count axis (can be negative for end triggers).

    Returns
    -------
    imv : pyadisi.pyqtgraph.imageview.ImageView.ImageView
        A neat viewer to investigate your image stack.
    data : dict
        The digitized locations, where keys are the frame number
        and the (x, y) values are in a list.
    proxy_chair : pyadisi.pyqtgraph.SignalProxy
        Signal proxy for the crosshairs.
    proxy_click : pyadisi.pyqtgraph.SignalProxy
        Signal proxy for the mouse click events (how we digitize).
    """

    imv = pg.ImageView()
    imv.setWindowTitle('pyadisi is cool')

    # we are not ready to use these yet, wo we hide them :)
    imv.ui.roiBtn.hide()
    imv.ui.normBtn.hide()

    # fix quickMinMax...maybe
    # We 'fix' this function so that we don't
    # need a numpy array...This is only called if
    # the images passed are not numpy ndarry.
    if not isinstance(images, np.ndarray):
        imv.quickMinMax = lambda x: (np.iinfo(x.dtype).min, np.iinfo(x.dtype).max)

    # parameters to setImage (if we have a video, single color, single gray scale)
    if images.ndim == 3 or images.ndim == 4:
        if xvals is None:
            xvals = np.arange(images.shape[0])
        axes = {'t':0, 'x':1, 'y':2, 'c':3}
    #elif images.ndim == 2:
    #    xvals = np.arange(images.shape[0])
    #    axes = {'x': 0, 'y': 1, 'c': 2}
    elif images.ndim == 2:
        xvals = None
        axes = {'x': 0, 'y': 1}

    # after it is doctored up, give it some images
    imv.setImage(images, xvals=xvals, axes=axes, autoHistogramRange=True)

    # we want to show the frame number on the image (eventually...)
    vb = imv.getView()
    label = pg.LabelItem(justify='right')
    #vb.addItem(label)
    label.setText("<span style='font-size: 26pt'>frame = {0}".format(imv.currentIndex))

    # finally show (not sure when we have to do this)
    imv.show()

    # we store the data in a dictionary
    data = {}
    def fill_data(current_index, point):
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
            #print('frame {2:4d} :   (x, y): ({0:.5f}, {1:.5f})'.format(mousePoint.x(), mousePoint.y(), imv.currentIndex))
            #sys.stdout.flush()

            # push the data into the dictionary
            fill_data(imv.currentIndex, (mousePoint.x(), mousePoint.y()))

    proxy_click = pg.SignalProxy(imv.scene.sigMouseClicked, rateLimit=60, slot=mouseClicked)


    if crosshair:
        # cross hair as intersection of two infinite lines
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