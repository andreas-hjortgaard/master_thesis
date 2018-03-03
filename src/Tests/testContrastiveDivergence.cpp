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
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "Types.h"
#include "DataManager.h"
#include "ObjectiveFunctions/LogLikelihood.h"
#include "ObjectiveFunctions/SampledGradient.h"
#include "Learning/ContrastiveDivergence.h"
#include "Learning/LBFGS.h"

using namespace std;

//int testStochasticGradient() {
int main(int argc, char **argv) {
  
  // initialize data manager
  DataManager dataman;
    
  try {
    dataman.loadImages("../cows-train/EUCSURF-3000/", "../subsets/cows_train_width_height.txt");
    dataman.loadBboxes("../cows-train/Annotations/TUcow_train.ess");
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
  Weights w(numWeights, 0.1);  
  
  // initialize weights between -1 and 1
  //for (int i=0; i<numWeights; i++) {
  //  w[i] = 0.1;
  //}

  // set weights  
  int stepSize = 16;
  crf.setStepSize(stepSize);
  crf.setWeights(w);

  // create log-likelihood objective function and gradient with regularization
  LogLikelihood loglik(&dataman, &crf);
  loglik.setLambda(2.0);

  LogLikelihoodGradient loglikgrad(&dataman, &crf);
  loglikgrad.setLambda(2.0);

  GibbsSampler gibbs(&crf);
  SampledGradient samplgrad(&dataman, &crf, &gibbs);
  samplgrad.setLambda(2.0);
  
  
  // test evaluate
  printf("Computing log-likelihood...\n");
  double fval   = loglik.evaluate(w);
  printf("Loglikelihood = %.6f\n", fval);

  //printf("Computing gradient...\n");
  //Dvector grad(num_weights);
  //loglik.gradient(grad, w);
  //printf("Gradient = (%.2f, %.2f, %.2f, ...)\n", grad[0], grad[1], grad[2]);
  
  
  // train parameters with CD
  // alpha and t0 are set to default
  int    k      = 1;
  double alpha  = 1;
  double t0     = 0; 
  ContrastiveDivergence cd(&loglik, &samplgrad, k, alpha, t0);
  LBFGS lbfgs(&loglik, &loglikgrad);
  Weights w_new(numWeights, 0.0);

  double start, stop, inittime, cdtime, lbfgstime, totaltime;
  double cdval;
  
  printf("Learning parameters with CD...\n");
  try {
    // initialize learning rate parameters
    start = gettime();
    cd.initializeLearningRate(w, 0.1, 0, false);
    stop = gettime();
    inittime = stop-start;    

    // store CD parameters
    ofstream params_file("cd_params.txt");
    if (!params_file) {
      cerr << "Could not open file cd_params.txt" << endl;
    }    
    params_file << "alpha: " << cd.getAlpha() << endl;
    params_file << "t0: " << cd.getT0() << endl;
 
    params_file.close();
    
    start = gettime();
    w_new = cd.learnWeights(w);
    stop = gettime();
    cdtime = stop-start;
    cdval = loglik.evaluate(w_new);  

    start = gettime();
    w_new = lbfgs.learnWeights(w_new);    
    stop = gettime();
    lbfgstime = stop-start;

    printf("Initial objective: %.6f\n", fval);
    fval = loglik.evaluate(w_new);
    printf("Final objective: %.6f\n\n", fval); 
    
    totaltime = inittime + cdtime + lbfgstime;
    
    ofstream cdTimeFile("cd_time.txt"); 
    if (!cdTimeFile) {
      cerr << "Could not open file cd_time.txt" << endl;
    }   
    cdTimeFile << "Initialization: " <<   inittime << endl;
    cdTimeFile << "CD:             " <<   cdtime << endl;
    cdTimeFile << "LBFGS:          " <<   lbfgstime << endl;
    cdTimeFile << "Total:          " <<   totaltime << endl << endl;
    cdTimeFile << "CD objective:    " << cdval << endl;
    cdTimeFile << "Final objective: " << fval << endl;
    
    cdTimeFile.close();

  } 
  catch (int e) {
    if (e == ROUNDOFF_ERROR) {
      fprintf(stderr, "CD::learnWeights threw a round-off error!\n");
      return ROUNDOFF_ERROR;
    }
    
    if (e == DIM_ERROR) {
      fprintf(stderr, "CD::learnWeights threw a dimensionality error!\n");
      return DIM_ERROR;
    }

    if (e == NOT_A_NUMBER) {
      fprintf(stderr, "CD::learnWeights threw a NAN error!\n");
      return NOT_A_NUMBER;
    }
  }
  
  
  ofstream weight_file("cd_weights_bicycle.txt");
  if (!weight_file) {
    cerr << "Could not open file weights.txt" << endl;
  }
  
  // store result in file
  for (int i=0; i<numWeights; i++) {
    weight_file << w_new[i] << endl;
  }

  weight_file.close();
  
  cout << "Done!" << endl;
  
  return 0;
}
