#!/usr/bin/python

import sys
	
import numpy as N
import os
import pylab 
from PIL import Image
from ctypes import c_uint8

try:
	w = pylab.mlab.load(sys.argv[1])
except IndexError:
        print "Usage: %s weight-file weights.txt image1.clst [image2.clst [...]]" % sys.argv[0]
	raise SystemExit

for xycname in sys.argv[2:]:

    # if file extension is clst, read ASCII
    ext = os.path.splitext(xycname)[1]
    if (ext == ".clst"):
      print "Using ASCII features..."
      xyc = pylab.mlab.load(xycname)
      
      
      
    else:
      # else read binary
      print "Using binary features"
      # get size and compute num features
      filesize = os.path.getsize(xycname)
      numfeatures = filesize/(3*2)
      print numfeatures
      shape=(3,numfeatures)
      
      xyc = N.fromfile(file=xycname, dtype=N.short).reshape(shape)
      xyc = xyc.T

    x = xyc[:,0]
    y = xyc[:,1]
    c = xyc[:,2]  

    
    maxx = int(max(x)+min(x))+1 # assumes that left and right border are of equal size
    maxy = int(max(y)+min(y))+1
    
    a = N.zeros( (maxy, maxx) , float )
    for x,y,c in xyc:
        a[ int(y), int(x) ] += w[int(c)]

    from scipy.signal import gaussian,convolve2d
    s=4
    gauss1d = N.reshape( gaussian(6*s,s), (-1,1))
    a = convolve2d(a,gauss1d, mode='same')
    a = convolve2d(a,gauss1d.T, mode='same')
    maxa = max( [N.max(a),0] )
    mina = min( [N.min(a),0] )
    minmaxa = max(maxa,abs(mina))
    a[a<0] /= abs(mina)
    a[a>0] /= maxa

    color_a = pylab.cm.jet((a+1)/2)
    color_a = N.array(255*color_a, c_uint8)
    im = Image.fromarray(color_a[:,:,:3])
    im.save("%s.jpg" % xycname)

