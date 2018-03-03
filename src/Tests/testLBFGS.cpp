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

#include <cstdio>
#include <iostream>
#include <fstream>

#include "DataManager.h"
#include "ConditionalRandomField.h"
#include "ObjectiveFunctions/LogLikelihood.h"
#include "ObjectiveFunctions/LogLikelihoodGradient.h"
#include "Learning/LBFGS.h"
#include "Types.h"

using namespace std;

//int testLBFGS() {
int main(int argc, char **argv) {
  
  // initialize data manager
  DataManager dataman;
    
  try {
    //dataman.loadImages("../pascal/USURF3K/", "subsets/train_width_height_bicycle.txt");
    //dataman.loadBboxes("../pascal/Annotations/ess/bicycle_train_onlyboxes.ess");
    dataman.loadImages("../cows-train/EUCSURF-3000/", "../subsets/cows_train10_width_height.txt");
    dataman.loadBboxes("../cows-train/Annotations/TUcow_train10.ess");
  }
  catch (int e) {
    if (e == FILE_NOT_FOUND) {
      fprintf(stderr, "There was a problem opening a file!\n");
      return -1;
    }
  }   
  
  // create conditional random field
  ConditionalRandomField crf(&dataman);
  //Weights w = dataman.getWeights();
  int numWeights = 3000;
  Weights w(numWeights);  
  
  // initialize weights between -1 and 1
  for (int i=0; i<numWeights; i++) {
    w[i] = 0.5;
  }

  // set weights  
  int stepSize = 32;
  crf.setStepSize(stepSize);
  crf.setWeights(w);

  // create log-likelihood objective function and gradient with regularization
  LogLikelihood loglik(&dataman, &crf);
  loglik.setLambda(2.0);

  LogLikelihoodGradient loglikgrad(&dataman, &crf);
  loglikgrad.setLambda(2.0);
  
  // train parameters with BFGS
  LBFGS lbfgs(&loglik, &loglikgrad);
  Weights w_new(numWeights, 0.0);
  
  printf("Learning parameters with LBFGS...\n");
  try {
    w_new = lbfgs.learnWeights(w);
  } 
  catch (int e) {
    if (e == ROUNDOFF_ERROR) {
      fprintf(stderr, "LBFGS::learnWeights threw a round-off error!\n");
      return ROUNDOFF_ERROR;
    }
    
    if (e == DIM_ERROR) {
      fprintf(stderr, "LBFGS::learnWeights threw a dimensionality error!\n");
      return DIM_ERROR;
    }

    if (e == NOT_A_NUMBER) {
      fprintf(stderr, "LBFGS::learnWeights threw a NAN error!\n");
      return NOT_A_NUMBER;
    }
  }
  
  
  ofstream weight_file("lbfgs_weights_bicycle.txt");
  if (!weight_file) {
    cerr << "Could not open file lbfgs_weights_bicycle.txt" << endl;
  }
  
  // store result in file
  for (int i=0; i<numWeights; i++) {
    weight_file << w_new[i] << endl;
  }

  weight_file.close();
  
  cout << "Done!" << endl;

  return 0;
}
