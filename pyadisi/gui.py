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

pg.setConfigOptions(antialias=True)


imv = pg.ImageView()

imv.setWindowTitle('pyadisi is cool')

imv.show()
imv.setImage(d, xvals=np.arange(d.shape[0]))

#label = pg.LabelItem(justify='right')


#cross hair
vLine = pg.InfiniteLine(angle=90, movable=False)
hLine = pg.InfiniteLine(angle=0, movable=False)
imv.addItem(vLine, ignoreBounds=True)
imv.addItem(hLine, ignoreBounds=True)

vb = imv.getView()

def mouseMoved(evt):
    pos = evt[0]  ## using signal proxy turns original arguments into a tuple
    if imv.getImageItem().sceneBoundingRect().contains(pos):
        mousePoint = vb.mapSceneToView(pos)
        #index = int(mousePoint.x())
        #if index > 0 and index < len(data1):
        #    label.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), data1[index], data2[index]))
        vLine.setPos(mousePoint.x())
        hLine.setPos(mousePoint.y())

def mouseClicked(evt):
    mouseclick = evt[0]
    pos = mouseclick.scenePos().toQPoint()  # https://github.com/pyqtgraph/pyqtgraph/blob/develop/pyqtgraph/Point.py#L154
    in_scene = imv.getImageItem().sceneBoundingRect().contains(pos)
    if in_scene and mouseclick.button() == 1:
        # .contains() requires a QtCore.QPointF, but we get a Point (subclassed from QtCore.QPointF) from the event
        mousePoint = vb.mapSceneToView(pos)
        print('frame {2:4d} :   (x, y): ({0:.5f}, {1:.5f})'.format(mousePoint.x(), mousePoint.y(), imv.currentIndex))
        sys.stdout.flush()

proxy_chair = pg.SignalProxy(imv.scene.sigMouseMoved, rateLimit=60, slot=mouseMoved)
proxy_click = pg.SignalProxy(imv.scene.sigMouseClicked, rateLimit=60, slot=mouseClicked)