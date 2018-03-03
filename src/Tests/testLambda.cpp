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
#include <cmath>

#include "Types.h"
#include "DataManager.h"
#include "ConditionalRandomField.h"
#include "PiecewiseConditionalRandomField.h"
#include "Inference/GibbsSampler.h"
#include "ObjectiveFunctions/LogLikelihood.h"
#include "ObjectiveFunctions/PseudoLikelihood.h"
#include "ObjectiveFunctions/PiecewiseLogLikelihood.h"
#include "ObjectiveFunctions/LogLikelihoodGradient.h"
#include "ObjectiveFunctions/StochasticGradient.h"
#include "ObjectiveFunctions/SampledGradient.h"
#include "ObjectiveFunctions/PseudoLikelihoodGradient.h"
#include "ObjectiveFunctions/PiecewiseGradient.h"
#include "Learning/LBFGS.h"

using namespace std;

int main(int argc, char **argv) {
  
  // initialize data manager
  DataManager dataman;
    
  try {
    dataman.loadImages("../pascal/USURF3K/", "../subsets/train_width_height.txt");
    dataman.loadBboxes("../pascal/Annotations/ess/cat_train.ess");
  }
  catch (int e) {
    if (e == FILE_NOT_FOUND) {
      fprintf(stderr, "There was a problem opening a file!\n");
      return -1;
    }
  }
  
  // create conditional random field
  ConditionalRandomField crf(&dataman);
  PiecewiseConditionalRandomField pcrf(&dataman);
  //Weights w = dataman.getWeights();
  int weightDim = 3000;
  Weights w(weightDim, 0.1);  
  
  // initialize weights between -1 and 1
  srand(time(NULL)); rand();
  for (int i=0; i<weightDim; i++) {
    w[i] = ((double) rand() / RAND_MAX)*20 - 10;
  }

  // set weights  
  int stepSize = 32;
  crf.setStepSize(stepSize);
  crf.setWeights(w);
  pcrf.setStepSize(stepSize);
  pcrf.setWeights(w);
  
  // create Gibbs sampler
  GibbsSampler gibbs(&crf);

  // create log-likelihood objective function and gradient with regularization
  LogLikelihood loglik(&dataman, &crf);
  PseudoLikelihood ploglik(&dataman, &crf);
  PiecewiseLogLikelihood pwloglik(&dataman, &pcrf);
  LogLikelihoodGradient loglikgrad(&dataman, &crf);
  StochasticGradient stochgrad(&dataman, &crf);
  SampledGradient samplgrad(&dataman, &crf, &gibbs);
  PseudoLikelihoodGradient plikgrad(&dataman, &crf);
  PiecewiseGradient pwlikgrad(&dataman, &pcrf);
  
  samplgrad.setNumSamples(1);
  samplgrad.setSkip(1);
  
  
  // try different values of lambda
  double min = 1e-8;
  double max = 1e8;
  double fval, pval, pwval, gnorm, sgnorm, pgnorm, pwgnorm;
  Dvector grad(weightDim);
  Dvector smplgrad(weightDim);
  Dvector pgrad(weightDim);
  Dvector pwgrad(weightDim);
  
  for (double lam=min; lam<=max; lam *= 10.0) {
    loglik.setLambda(lam);
    ploglik.setLambda(lam);
    pwloglik.setLambda(lam);
    
    loglikgrad.setLambda(lam); 
    stochgrad.setLambda(lam);
    samplgrad.setLambda(lam);
    plikgrad.setLambda(lam);
    pwlikgrad.setLambda(lam);
    
    // objective functions
    fval = loglik.evaluate(w);
    pval = ploglik.evaluate(w);
    pwval = pwloglik.evaluate(w);
    
    // gradients
    stochgrad.evaluate(grad, w, 0);
    samplgrad.evaluate(smplgrad, w, 0);
    plikgrad.evaluate(pgrad, w);
    pwlikgrad.evaluate(pwgrad, w);
    
    gnorm = 0.0;
    sgnorm = 0.0;
    pgnorm = 0.0;
    pwgnorm = 0.0;
    for (size_t i=0; i<grad.size(); i++) {
      gnorm = grad[i]*grad[i];
      sgnorm = smplgrad[i]*smplgrad[i];
      pgnorm = pgrad[i]*pgrad[i];
      pwgnorm = pwgrad[i]*pwgrad[i];
    }
    gnorm = sqrt(gnorm);
    sgnorm = sqrt(sgnorm);
    pgnorm = sqrt(pgnorm);
    pwgnorm = sqrt(pwgnorm);
    
    cout << "Lambda:                " << lam << endl;
    cout << "Loglikelihood:         " << fval << endl;
    cout << "PseudoLikelihood:      " << pval << endl;
    cout << "PiecewiseLikelihood:   " << pwval << endl; 
    cout << "Stochastic norm:       " << gnorm << endl;
    cout << "Sampled norm:          " << sgnorm << endl;
    cout << "PseudoGrad norm:       " << pgnorm << endl;
    cout << "PiecewiseGrad norm:    " << pwgnorm << endl << endl;
    
  } 
  
  cout << "Done!" << endl;
  
  return 0;
}
