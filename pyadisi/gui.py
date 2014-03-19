"""GUI stuff for dealing with videos and tracking.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets
from matplotlib.image import AxesImage

from scipy import ndimage
from scipy.ndimage.measurements import center_of_mass
from skimage.filter import threshold_otsu
from skimage.exposure import adjust_gamma

import sys



class CamPicker2D(object):
    """Pick points out from images.
    """

    def __init__(self, cams, nframes, npoints, start_frame=0, color='white'):

        cam1, cam2 = cams
        self.cam1 = cam1
        self.cam2 = cam2
        ncams = len(cams)
        self.cams = cams
        self.ncams = ncams
        self.nframes = nframes
        self.npoints = npoints

        data_tmp = np.zeros((npoints, nframes, 2)) * np.nan  # point, frame num, x and y
        self.data = dict(cam1=data_tmp.copy(), cam2=data_tmp.copy())

        self.frame_num = start_frame
        self.point_num = 0

        self.setup_figure()

    def setup_figure(self):

        #cam1, cam2 = self.cam1, self.cam2
        self.fig = plt.figure(figsize=(8, 6))
        self.ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
        self.ax2 = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=2)
        self.ax3 = plt.subplot2grid((4, 4), (2, 0))
        self.ax4 = plt.subplot2grid((4, 4), (2, 1))
        self.ax5 = plt.subplot2grid((4, 4), (2, 2))
        self.ax6 = plt.subplot2grid((4, 4), (2, 3))
        self.axs = [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]

        for ax in self.axs:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        self.imshow_args = dict(interpolation='nearest', cmap=plt.cm.gray)
        self.p1 = self.ax1.imshow(self.cam1[self.frame_num], **self.imshow_args)
        self.p2 = self.ax2.imshow(self.cam2[self.frame_num], **self.imshow_args)

        ## fill-up the small axes
        tmp_data = np.zeros((16, 16))
        self.p3 = self.ax3.imshow(tmp_data, **self.imshow_args)
        self.p4 = self.ax4.imshow(tmp_data, alpha=.8, **self.imshow_args)
        self.p5 = self.ax5.imshow(tmp_data, **self.imshow_args)
        self.p6 = self.ax6.imshow(tmp_data, alpha=.8, **self.imshow_args)

        self.com_args = dict(ms=12, alpha=.95)
        com = np.array([8, 8])
        self.cmplot3, = self.ax3.plot(com[1], com[0], 'g+', **self.com_args)
        self.cmplot4, = self.ax4.plot(com[1], com[0], 'g+', **self.com_args)
        self.cmplot5, = self.ax5.plot(com[1], com[0], 'g+', **self.com_args)
        self.cmplot6, = self.ax6.plot(com[1], com[0], 'g+', **self.com_args)

        self._cmtxt = '{0:.2f}, {1:.2f}'
        self._cmtxt_args = dict(color='r', fontsize=12)
        self.cmtxt4 = self.ax4.text(1, 14, self._cmtxt.format(com[1], com[0]), **self._cmtxt_args)
        self.cmtxt6 = self.ax6.text(1, 14, self._cmtxt.format(com[1], com[0]), **self._cmtxt_args)

        self.ax3.axis('image')
        self.ax4.axis('image')
        self.ax5.axis('image')
        self.ax6.axis('image')

        self._txt = 'frame {0:03d}, point {1:d}'
        self.t1 = self.ax1.set_title(self._txt.format(self.frame_num, self.point_num), fontsize=14)
        self.t2 = self.ax2.set_title(self._txt.format(self.frame_num, self.point_num), fontsize=14)

        self.fig.tight_layout()
        self.fig.set_facecolor('w')

        cursor_args = dict(useblit=True, color='red', linewidth=1, alpha=.8)
        self.cursor1 = widgets.Cursor(self.ax1, **cursor_args)
        self.cursor2 = widgets.Cursor(self.ax2, **cursor_args)
        #self.cursor34 = widgets.MultiCursor(self.fig.canvas, (self.ax3, self.ax4), color='r', lw=1, ls='--',
        #            horizOn=True, vertOn=True)
        #self.cursor56 = widgets.MultiCursor(self.fig.canvas, (self.ax5, self.ax6), color='r', lw=1, ls='--',
        #            horizOn=True, vertOn=True)

        self._ax_slider = plt.axes([0.09, 0.15, 0.375, 0.025])
        self.pix_slider = widgets.Slider(self._ax_slider, 'select', 2, 30, valinit=10, color='#AAAAAA')

        self._ax_gamma = plt.axes([.09, .075, 0.375, .025])
        self.gam_slider = widgets.Slider(self._ax_gamma, 'gamma', 0, 2, valinit=1, color='#AAAAAA')#, picker=agamma)

        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        #fig.canvas.mpl_connect('pick_event', self.on_pick)

    def _pix(self):
        return np.round(self.pix_slider.val).astype(np.int)

    def _yloc_cm(self):
        return 2 * self._pix() - 2

    def on_scroll(self, event):
        button = event.button
        if button == 'down':
            self.frame_num += 1
            if self.frame_num > self.nframes:
                self.frame_num = self.nframes - 1
        if button == 'up':
            self.frame_num -= 1
            if self.frame_num < 0:
                self.frame_num = 0
        self.update()

    def on_pick(self, event):
        print 'here in on_pick'
        self.event = event
        sys.stdout.flush()

    def on_key_press(self, event):
        key = event.key

        if key == 'l' or key == 'right':
            self.frame_num += 1
            if self.frame_num > self.nframes:
                self.frame_num = self.nframes - 1
        if key == 'h' or key == 'left':
            self.frame_num -= 1
            if self.frame_num < 0:
                self.frame_num = 0
        if key == 'n' or key == 'w':
            self.point_num += 1
            if self.point_num > self.npoints - 1:
                self.point_num = self.npoints - 1
        if key == 'p' or key == 'q':
            self.point_num -= 1
            if self.point_num < 0:
                self.point_num = 0

        self.update()

    def update(self):
        """Update the plots and data structures.
        """
        self.p1.set_data(self.cam1[self.frame_num])
        self.p2.set_data(self.cam2[self.frame_num])
        self.t1.set_text(self._txt.format(self.frame_num, self.point_num))
        self.t2.set_text(self._txt.format(self.frame_num, self.point_num))

        plt.draw()
        #fig.canvas.update()
        sys.stdout.flush()


    def on_button_press(self, event):

        # check if using zoom or pan tool; stackoverflow.com/q/20711148
        if not self.fig.canvas.manager.toolbar._active == None:
            return

        self.event = event

        if event.inaxes == self.ax1:
            x, y = event.xdata, event.ydata
            pix = self._pix()
            img = self.cam1[self.frame_num]
            subimg = img[y-pix:y+pix, x-pix:x+pix]  #TODO
            subimg_bin = (subimg > threshold_otsu(subimg)).astype(np.int)

            com = center_of_mass(subimg_bin)
            xc = com[1] + x - pix
            yc = com[0] + y - pix
            point = [yc, xc]

            # store point location
            self.data['cam1'][self.point_num, self.frame_num, :] = point

            self.ax3.clear()
            self.ax4.clear()
            self.ax3.imshow(subimg, **self.imshow_args)
            self.ax4.imshow(subimg_bin, alpha=.8, **self.imshow_args)
            #self.p3.set_array(subimg, **self.imshow_args); self.cmplot3.set_data(com)

            self.ax3.plot(com[1], com[0], 'g+', **self.com_args)
            self.ax4.plot(com[1], com[0], 'g+', **self.com_args)
            #TODO use this as a way to store which points have been recorded
            #TODO a colored matrix (and click on points, frames yet to be analyzed?)
            #self.cmtxt4.set_text(self._cmtxt.format(yc, xc))  #TODO why is this not updating?
            self.cmtxt4 = self.ax4.text(1, self._yloc_cm(), self._cmtxt.format(yc, xc), **self._cmtxt_args)

            self.ax3.axis('image')
            self.ax4.axis('image')

        if event.inaxes == self.ax2:
            x, y = event.xdata, event.ydata
            pix = self._pix()
            img = self.cam2[self.frame_num]
            subimg = img[y-pix:y+pix, x-pix:x+pix]  #TODO
            subimg_bin = (subimg > threshold_otsu(subimg)).astype(np.int)

            com = center_of_mass(subimg_bin)
            xc = com[1] + x - pix
            yc = com[0] + y - pix
            point = [yc, xc]

            # store point location
            self.data['cam2'][self.point_num, self.frame_num, :] = point

            self.ax5.clear()
            self.ax6.clear()
            self.ax5.imshow(subimg, **self.imshow_args)
            self.ax6.imshow(subimg_bin, alpha=.8, **self.imshow_args)

            self.ax5.plot(com[1], com[0], 'g+', **self.com_args)
            self.ax6.plot(com[1], com[0], 'g+', **self.com_args)
            #self.cmtxt6.set_text(self._cmtxt.format(yc, xc))
            self.cmtxt6 = self.ax6.text(1, self._yloc_cm(), self._cmtxt.format(yc, xc), **self._cmtxt_args)

            self.ax5.axis('image')
            self.ax6.axis('image')

        if event.inaxes == self.ax3 or event.inaxes == self.ax4:
            #TODO since we update the point each time, and not the com, this will start to do
            #weird things; make it so they can only click once?
            xc, yc = self.data['cam1'][self.point_num, self.frame_num, :]
            x, y = event.xdata, event.ydata
            comx = xc - x + self._pix()
            comy = yc - y + self._pix()

            self.ax3.plot(x, y, 'r+', **self.com_args)
            self.ax4.plot(x, y, 'r+', **self.com_args)

            com = np.array([comx, comy])
            self.data['cam1'][self.point_num, self.frame_num, :] = com
            #TODO to see this, uncomment below
            #print com

        if event.inaxes == self.ax5 or event.inaxes == self.ax6:
            #TODO since we update the point each time, and not the com, this will start to do
            #weird things; make it so they can only click once?
            xc, yc = self.data['cam2'][self.point_num, self.frame_num, :]
            x, y = event.xdata, event.ydata
            comx = xc - x + self._pix()
            comy = yc - y + self._pix()

            self.ax5.plot(x, y, 'r+', **self.com_args)
            self.ax6.plot(x, y, 'r+', **self.com_args)

            com = np.array([comx, comy])
            self.data['cam2'][self.point_num, self.frame_num, :] = com
            #TODO to see this, uncomment below
            #print com

        sys.stdout.flush()
        plt.draw()