#!/bin/bash

objects=("bicycle" "bus" "car" "cat" "cow" "dog" "horse" "motorbike" "person" "sheep")

# pseudolikelihood
#lambdas=(10000 1000 10000 10000 10000 10000 10000 10000 10000 10000)
#for i in {0..9}; do
#  /home/danielsen/Documents/spod/src/Exec/testPerformance /home/danielsen/Documents/spod/ ${objects[$i]} 8 ${lambdas[$i]} pseudolikelihood /home/danielsen/Desktop/results/pseudolikelihood/${objects[$i]}_8_${lambdas[$i]}_weights.txt 
#done;

# piecewise
#lambdas=(10000 10000 10000 100000 10000 10000 10000 100000 100000 100000)
#for i in {0..9}; do
#  /home/danielsen/Documents/spod/src/Exec/testPerformance /home/danielsen/Documents/spod/ ${objects[$i]} 8 ${lambdas[$i]} piecewise /home/danielsen/Desktop/results/piecewise/${objects[$i]}_8_${lambdas[$i]}_weights.txt
#done;

# cd
#lambdas=(100 10 10 100 100 10 100 10 100 100)
#for i in {0..9}; do
#  /home/danielsen/Documents/spod/src/Exec/testPerformance /home/danielsen/Documents/spod/ ${objects[$i]} 8 ${lambdas[$i]} cd /home/danielsen/Desktop/results/cd/${objects[$i]}_8_${lambdas[$i]}_200_1_weights.txt
#done;


# structured svm
#for i in {0..9}; do
#  /home/danielsen/Documents/spod/src/Exec/testPerformance /home/danielsen/Documents/spod/ ${objects[$i]} 1 0 ssvm /home/danielsen/Documents/spod/pascal/Weights/${objects[$i]}2006-w.txt
#done;


# random [-0.1, 0.1]
#for i in {0..9}; do
#  /home/danielsen/Documents/spod/src/Exec/testPerformance /home/danielsen/Documents/spod/ ${objects[$i]} 1 0 rand0.1 /home/danielsen/Documents/spod/src/weights/random_weights/randWeights_3000_0.1.txt
#done;


# random [-0.0001, 0.0001]
#for i in {0..9}; do
#  /home/danielsen/Documents/spod/src/Exec/testPerformance /home/danielsen/Documents/spod/ ${objects[$i]} 1 0 rand0.0001 /home/danielsen/Documents/spod/src/weights/random_weights/randWeights_3000_0.0001.txt
#done;



## TUCOW

# lbfgs
#/home/danielsen/Documents/spod/src/Exec/testPerformance /home/danielsen/Documents/spod/ tucow 8 1000 lbfgs /home/danielsen/Documents/spod/cluster/results_tucow_trainval/lbfgs/tucow_8_1000_weights.txt 

# pseudolikelihood
#/home/danielsen/Documents/spod/src/Exec/testPerformance /home/danielsen/Documents/spod/ tucow 8 100 pseudolikelihood /home/danielsen/Documents/spod/cluster/results_tucow_trainval/pseudolikelihood/tucow_8_100_weights.txt 

# piecewise
#/home/danielsen/Documents/spod/src/Exec/testPerformance /home/danielsen/Documents/spod/ tucow 8 1000 piecewise /home/danielsen/Documents/spod/cluster/results_tucow_trainval/piecewise/tucow_8_1000_weights.txt 

# sgd
/home/danielsen/Documents/spod/src/Exec/testPerformance /home/danielsen/Documents/spod/ tucow 8 10 sgd /home/danielsen/Documents/spod/cluster/results_tucow_trainval/sgd/tucow_8_10_50_weights.txt 

# cd
/home/danielsen/Documents/spod/src/Exec/testPerformance /home/danielsen/Documents/spod/ tucow 8 100 cd /home/danielsen/Documents/spod/cluster/results_tucow_trainval/cd/tucow_8_100_200_1_weights.txt 
