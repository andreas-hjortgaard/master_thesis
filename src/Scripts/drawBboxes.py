import Image
import ImageDraw
import sys
import os

# usage: drawBboxes input output x1 y1 x2 y2 x1 y1 x2 y2 ... 


inputfile   = sys.argv[1]
outputfile  = sys.argv[2]

numboxes = (len(sys.argv)-2)/4

print numboxes

im = Image.open(inputfile)

for i in range(0,numboxes):
  index = 4*i+3
  x1 = int(sys.argv[index])
  y1 = int(sys.argv[index+1])
  x2 = int(sys.argv[index+2])
  y2 = int(sys.argv[index+3])
  
  
  draw = ImageDraw.Draw(im)
  draw.rectangle([x1,y1,x2,y2])
  

del draw
#fileName, fileExtension = os.path.splitext(imagefile)
#out = "%s_Bbox%s" % (fileName, fileExtension)
#print out

im.save(outputfile)
