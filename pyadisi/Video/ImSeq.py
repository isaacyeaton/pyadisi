#ImageSequence Related classes/functions

#TODO
## Make loading/saving image sequences better (it sucks right now)


import numpy as np
import sys
import os
from pylab import imread
import re
import cv2
import time
import cPickle as pik

class ImSeq:
    """An easy way to access all the images in a directory"""
    def __init__(self, path, basename, loop=False, wantgray=False, note=''):
        self.path = path
        if self.path[-1] is not os.sep:
            self.path += os.sep #path ends in / for python searching purposes
        self.list = self.get_list()
        self.index = 0
        self.loop = loop # when going prev/next, will the im seg end
        self.basename = basename 
        self.wantgray= wantgray #if True, return grayscale images when fetched
        #self.eq = eq
        self.num_frames = len(self.list)
        self.isgray = self._check_gray() #are the images we are loading inherantly grayscale?
        self.wantgray = self.wantgray or self.isgray #if it is gray, we have no choice
        self.avg = None
        self._max = 0
        self._min = 0
        self._get_stats()
        self.note = note #string info from saving.etc

    def __repr__(self):
        return "FrogFrames Object from %s" %(self.path)
    def __str__(self):
        return self.__repr__()   
    
    def _get_stats(self):
        """want to calc min and max values of all frames so we can convert them to
        arbitrary bit depths/formats/etc"""
        max = 0
        min = 0
        for k in range(self.num_frames):
            im = self.get_frame(k, wantgray=True)
            if im.max() > max:
                max = im.max()
            if im.min() < min:
                min = im.min()
        self._min = min
        self._max = max
    
    def _check_gray(self):
        """Returns True if images are grayscale"""
        try:
            im = imread('%s%s' % (self.path, self.list[0]))
        except:
            im = cv2.imread('%s%s' % (self.path, self.list[0]))
        if len(im.shape) == 2:
            return True
        elif im.shape[2] == 1:
            return True
        else:
            return False


    def get_list(self):
        """Gives a list of all the images in chronological order"""    
        #os.chdir(self.path)
        dirList = os.listdir(self.path)
        #print dirList
        dirList.sort()
        basename = self.basename
        dirList = [ m.group(0) for m in (re.search(r"(basename\d+).(png|tif)", img) for img in dirList) if m]   
        if dirList == []:
            raise TypeError("%s is not a basename in given directory" % basename)
        #print dirList
        return dirList
        
    def get_next_im(self, wantgray=None):
        """Gives the next frame in the frog video frame directory.
        If not looping, will return None when the video is done"""
        #print self.list[self.index]
        if self.index >= len(self.list):
            #Check to see if current index is at the end of the sequence
            if self.loop: #if we are looping, go back to the beginning
                self.index = np.mod(self.index, len(self.list))
            else:
                return None, np.NaN    
        im = self.get_frame(self.index, wantgray)           
        
        self.index = self.index+1    
        
        return im, self.index-1

    def get_frame(self, num, wantgray=None):
        """Returns frame number <num> mod number of frames
        NOTE: first frame is frame 0"""
        num = np.mod(num, len(self.list))
        if wantgray is None: #Default to grayscale defined in beginning
            wantgray = self.wantgray

        GRAY = False

        if not self.isgray: # if its already gray, we dont need to do anything
            if wantgray:
                GRAY = True
        #print GRAY
        
        try:
            im = imread('%s%s' % (self.path, self.list[num]), flatten = GRAY)
            im = im.astype('uint16')
        except:
            tmpim = cv2.imread('%s%s' % (self.path, self.list[num]), not GRAY )
            if not self.isgray and not wantgray:
                im = tmpim.copy() #CV2 reads in BGR - its RGB now
                im[:,:,0] =tmpim[:,:,2]
                im[:,:,2] = tmpim[:,:,0]
            else: im = tmpim
  
        return im

    def calc_avg(self):
        sum_f = np.zeros(self.get_frame(0).shape)
        for k in range(self.num_frames):
            try:
                sum_f += self.get_frame(k)
            except:
                print "something went wrong on frame %d" %k
        sum_f /= self.num_frames

        self.avg = sum_f

    def get_mean_sub_frame(self, num):
        if self.avg is None:
            print "Need to calculate average first"
            self.calc_avg()
        im = self.get_frame(num)
        #cast to uint8 because fuck it

        im = abs(im.astype('float') - self.avg)
        #im = cv2.inRange(im, avg - 5, avg + 5)
        return im.astype('uint16')\

    #def get_median_im(self):
    #    #get the median image of the frog video
    #    winsize = 

    def save_imseq(self, filename, note='None'):
        """This will save an instance of the imagesequence, for loading using a separate function.
        It will also save a date that is was saved, and an optional note that will be printed along
        with the date.  This note is cumulative."""
        thenote = self.note + '\n' + time.asctime() + ':%s' %(note)

        dict = {path : self.path,
                imlist : self.list,
                index : self.index,
                loop : self.loop,
                basename : self.basename,
                numframes : self.num_frames,
                wantgray : self.wantgray,
                avg : self.avg,
                note : thenote}
        with open("%s.imseq" %(filename), 'w') as f:
            pik.dump(f, dict)
        return

def load_ImSeq(pik_file):
    """Given a pickled save instance of the ImSeq class, will reload the image seqeunce with note"""
    with open(pik_file, 'r') as f:
        k = pik.load(f)
    vid = ImSeq(path=k.path, basename = k.basename, loop=k.loop, wantgray=k.wantgray, note=k.note)    

    vid.avg = k.avg

    return vid