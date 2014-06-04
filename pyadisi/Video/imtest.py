import numpy as np
from pylab import imread


class ImTest():

	def __init__(self, path, basename):
		# dytpe, max, min, ndim, shape, size
		self.basename=basename
		self.path = path
        if self.path[-1] is not os.sep:
            self.path += os.sep #path ends in / for python searching purposes
        self.list = self.get_list()

        self.time_len = len(self.list)
        self.set_attributes()

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

    def set_attributes(self):
    	im = imread('%s%s' %(self.path, self.list[0]))
    	self.shape = tuple([self.time_len] + [k for k in im.shape])
    	self.dtype = im.dtype
    	self.ndim = len(self.shape)
    	self.size = reduce(lambda x, y: x*y, self.shape)

	def __getitem__(self, k):
		im = imread('%s%s' %(self.path, self.list[k]))
		return im