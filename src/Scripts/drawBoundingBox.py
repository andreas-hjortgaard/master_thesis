import Image
import ImageDraw
import sys
import os

imagefile = sys.argv[1]
x1 = int(sys.argv[2])
y1 = int(sys.argv[3])
x2 = int(sys.argv[4])
y2 = int(sys.argv[5])

thickness = 5
color = "green"

im = Image.open(imagefile)
draw = ImageDraw.Draw(im)

draw.rectangle([x1,y1,x1+thickness,y2], fill=color)   # left
draw.rectangle([x1,y1,x2,y1+thickness], fill=color)   # top
draw.rectangle([x2-thickness,y1,x2,y2], fill=color)   # right
draw.rectangle([x1,y2-thickness,x2,y2], fill=color)   # bottom
del draw 

fileName, fileExtension = os.path.splitext(imagefile)
out = "%s_Bbox%s" % (fileName, fileExtension)
print out
im.save(out)
