/* Conditional Random Fields for Object Localization
 * Master thesis source code
 * 
 * Authors:
 * Andreas Christian Eilschou (jwb226@alumni.ku.dk)
 * Andreas Hjortgaard Danielsen (gxn961@alumni.ku.dk)
 *
 * Department of Computer Science
 * University of Copenhagen
 * Denmark
 *
 * Date: 27-08-2012
 */

// test of computing losses of random weights and printing figures
#include <iostream>
#include <string>
#include <cstdlib>

#include "Measures/LossMeasures.h"

using namespace std;


int main(int argc, char **argv) {

  string plotName, pathWeights;
  string pathImages = "../pascal/USURF3K/";
  string pathSubset = "../subsets/train_width_height.txt";
  string pathBboxes = "../pascal/Annotations/ess/bicycle_train.ess";
  int stepSize = 16;
  
  // set random weights
  int num_weights = 3000;
  Weights w(num_weights);
  
  DataManager lossDataMan(pathImages, pathBboxes, pathSubset);
  RecallOverlap recallOverlap;
  
  int iterations = 100;
  
  double avgAUC = 0.;
  
  srand(time(NULL)); rand();
  for (int j = 0; j < iterations; j++) {
    
    if (j % 10 == 0) cout << "Iteration " << j << " out of " << iterations << endl;
    
    // initialize weights between -1 and 1
    for (int i=0; i<num_weights; i++) {
      w[i] = ((double) rand() / RAND_MAX)*0.2 - 0.1;
    }
    
    lossDataMan.setWeights(w);
    recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, false);
    avgAUC += recallOverlap.AUC/iterations;
  }
  
  cout << "Avg AUC: " << avgAUC << endl;
  
  plotName = "weights/loss_random_weights";
  printRecallOverlap(plotName, recallOverlap);
  
  return 0;
}
