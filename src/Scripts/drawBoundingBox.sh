#!/bin/bash
#
# Usage: ./drawBoundingBox.sh filename(not including path) x1 y1 x2 y2

python drawBoundingBox.py ../../cows-test/PNGImages/$1 $2 $3 $4 $5
eog ../../cows-test/PNGImages/$1 &

#python drawBoundingBox.py ../../pascal/PNGImages/$1 $2 $3 $4 $5
#eog ../../pascal/PNGImages/$1 &
