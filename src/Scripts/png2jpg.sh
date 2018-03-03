#!/bin/bash

files="./*.png"
for i in $files
do
  newname=$(echo "$i" | sed -e 's/\.png$/.jpg/')
  convert $i $newname
done
