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

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "DataManager.h"
#include "ConditionalRandomField.h"
#include "PiecewiseConditionalRandomField.h"
#include "ObjectiveFunctions/LogLikelihood.h"
#include "ObjectiveFunctions/LogLikelihoodGradient.h"
#include "ObjectiveFunctions/StochasticGradient.h"
#include "ObjectiveFunctions/SampledGradient.h"
#include "ObjectiveFunctions/PseudoLikelihood.h"
#include "ObjectiveFunctions/PseudoLikelihoodGradient.h"
#include "ObjectiveFunctions/PiecewiseLogLikelihood.h"
#include "ObjectiveFunctions/PiecewiseGradient.h"
#include "Inference/GibbsSampler.h"
#include "Types.h"

// INFERENCE TEST
using namespace std;
int main(int argc, char **argv) {

 // initialize data manager
  DataManager dataman;
    
  try {
    dataman.loadImages("../cows-train/EUCSURF-3000/", "../subsets/cows_train10_width_height.txt");
    dataman.loadBboxes("../cows-train/Annotations/TUcow_train10.ess");
    dataman.loadWeights("weights/sgd_weights_cows.txt");
  }
  catch (int e) {
    if (e == FILE_NOT_FOUND) {
      fprintf(stderr, "There was a problem opening a file!\n");
      return -1;
    }
  }
  
  // create log-likelihood objective function with regularization
  ConditionalRandomField crf(&dataman);
  PiecewiseConditionalRandomField pcrf(&dataman);
  Weights w = dataman.getWeights();
  //Weights w(3000, 0.0);  
  
  // set weights  
  int stepSize = 20;
  crf.setStepSize(stepSize);
  crf.setWeights(w);
  pcrf.setStepSize(stepSize);
  pcrf.setWeights(w);
  
  // create loglikelihood and gradients
  LogLikelihood loglik(&dataman, &crf);
  PseudoLikelihood pseudolik(&dataman, &crf);
  PiecewiseLogLikelihood piecewiselik(&dataman, &pcrf);
  
  LogLikelihoodGradient loglikgrad(&dataman, &crf);
  StochasticGradient stochgrad(&dataman, &crf);
  GibbsSampler gibbs(&crf);
  SampledGradient samplegrad(&dataman, &crf, &gibbs);
  PseudoLikelihoodGradient pseudograd(&dataman, &crf);
  PiecewiseGradient piecegrad(&dataman, &pcrf);

  // set lambda
  double lambda = 1000.;

  loglik.setLambda(lambda);
  pseudolik.setLambda(lambda);
  piecewiselik.setLambda(lambda);

  loglikgrad.setLambda(lambda);
  stochgrad.setLambda(lambda);
  samplegrad.setLambda(lambda);
  piecegrad.setLambda(lambda);


  // evaluate loglikelihood
  double loglikval, pseudolikval, piecewiseval;
  double start, stop;
  cout << "Evaluating loglikelihood..." << endl;
  start = gettime();
  loglikval = loglik.evaluate(w);
  stop = gettime();
  cout << "Val: " << loglikval << endl;
  cout << "Time taken: " << stop-start << " seconds" << endl << endl;

  // evaluate pseudolikelihood
  cout << "Evaluating pseudolikelihood..." << endl;
  start = gettime();
  pseudolikval = pseudolik.evaluate(w);
  stop = gettime();
  cout << "Val: " << pseudolikval << endl;
  cout << "Time taken: " << stop-start << " seconds" << endl << endl;

  // evaluate piecewiseloglikelihood
  cout << "Evaluating piecewise loglikelihood..." << endl;
  start = gettime();
  piecewiseval = piecewiselik.evaluate(w);
  stop = gettime();
  cout << "Val: " << piecewiseval << endl;
  cout << "Time taken: " << stop-start << " seconds" << endl << endl;


  // evaluate gradient
  Dvector gradient(w.size());    
  cout << "Evaluating gradient..." << endl;
  start = gettime();
  loglikgrad.evaluate(gradient, w);
  stop = gettime();
  cout << "Grad[0]: " << gradient[0] << endl;
  cout << "Time taken: " << stop-start << " seconds" << endl << endl;

  // evaluate stochastic gradient
  Dvector sgradient(w.size());    
  cout << "Evaluating stochastic gradient..." << endl;
  start = gettime();
  stochgrad.evaluate(gradient, w, 0);
  stop = gettime();
  cout << "StochGrad[0]: " << sgradient[0] << endl;
  cout << "Time taken: " << stop-start << " seconds" << endl << endl;
  
  // evaluate sampled gradient
  Dvector samplgradient(w.size());    
  cout << "Evaluating sampled gradient..." << endl;
  start = gettime();
  samplegrad.evaluate(samplgradient, w);
  stop = gettime();
  cout << "SampledGrad[0]: " << samplgradient[0] << endl;
  cout << "Time taken: " << stop-start << " seconds" << endl << endl;

  // evaluate contrastive divergence gradient
  Dvector cdgradient(w.size());
  cout << "Evaluating contrastive divergence gradient..." << endl;
  start = gettime();
  samplegrad.setSearchIx(0);
  samplegrad.evaluate(cdgradient, w, 0);
  stop = gettime();
  cout << "CDGrad[0]: " << cdgradient[0] << endl;
  cout << "Time taken: " << stop-start << " seconds" << endl << endl;

  // evaluate pseudolikelihood gradient
  Dvector pseudogradient(w.size());
  cout << "Evaluating pseudolikelihood gradient..." << endl;
  start = gettime();
  pseudograd.evaluate(pseudogradient, w, 0);
  stop = gettime();
  cout << "CDGrad[0]: " << pseudogradient[0] << endl;
  cout << "Time taken: " << stop-start << " seconds" << endl << endl;

  // evaluate pseudolikelihood gradient
  Dvector piecewisegradient(w.size());
  cout << "Evaluating piecewise loglikelihood gradient..." << endl;
  start = gettime();
  piecegrad.evaluate(piecewisegradient, w, 0);
  stop = gettime();
  cout << "PiecewiseGrad[0]: " << piecewisegradient[0] << endl;
  cout << "Time taken: " << stop-start << " seconds" << endl << endl;
  

  // check sampled gradient against real
  int countOff = 0;
  double gnorm = 0;
  double snorm = 0;
  for (size_t i=0; i<gradient.size(); i++) {
    if (abs(gradient[i]-samplgradient[i]) > 1) {
      countOff++;
    }
    gnorm += gradient[i]*gradient[i];
    snorm += samplgradient[i]*samplgradient[i];
  }
  gnorm = sqrt(gnorm);
  snorm = sqrt(snorm);

  printf("off: %d\n", countOff);
  printf("gradient norm = %.6f\nsamplegradient norm = %.6f\n", gnorm, snorm);
  
  cout << "Done!" << endl;
  
  return 0;
}
