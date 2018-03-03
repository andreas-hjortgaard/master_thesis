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

#include <cmath>
#include <cstdlib>
#include <algorithm>

#include "Types.h"
#include "StochasticGradientDescent.h"

using namespace std;


// implementation of the stochastic gradient descent method

// constructors (initialize SGD parameters)
StochasticGradientDescent::StochasticGradientDescent(ObjectiveFunction *obj, StochasticGradient *grad) : 
  GradientDescent(obj, grad), 
  gradient(grad),  
  constLearningRate(false),
  alpha(0.), 
  t0(0.), 
  maxEpochs(50), 
  epoch(0), 
  t(0),
  tempWeightsPath("tempWeightsSGD.txt") { }

StochasticGradientDescent::StochasticGradientDescent(ObjectiveFunction *obj, StochasticGradient *grad, double alpha_, double t0_, bool constLearningRate_) : 
  GradientDescent(obj, grad), 
  gradient(grad),
  constLearningRate(constLearningRate_),
  alpha(alpha_), 
  t0(t0_), 
  maxEpochs(50),
  epoch(0), 
  t(0),
  tempWeightsPath("tempWeightsSGD.txt") { }

// getters and setters
double StochasticGradientDescent::getAlpha() {
  return alpha;
}

void StochasticGradientDescent::setAlpha(double alpha_) {
  alpha = alpha_;
}

double StochasticGradientDescent::getT0() {
  return t0;
}

void StochasticGradientDescent::setT0(double t0_) {
  t0 = t0_;
}

void StochasticGradientDescent::setMaxEpochs(int maxEpochs_) {
  maxEpochs = maxEpochs_;
}

// set path for storing temporary weights
void StochasticGradientDescent::setTempWeightsPath(string path) {
  tempWeightsPath = path;
}

// try eta on sample from the training set using unnormalized objective
// run one epoch on subset and return resulting function value
double StochasticGradientDescent::tryEta(Weights &w, double eta, SearchIx &subset, bool normalized) {
  
  Dvector grad(w.size(), 0.0);

  for (size_t i=0; i<subset.size(); i++) {
      
    gradient->evaluate(grad, w, i, normalized);
   
    // update weights
    // w = w + eta*d = w - eta*gradient
    for (size_t j=0; j<w.size(); j++) {    
      w[j] = w[j] - eta*grad[j];
    }
  
  }
  return objective->evaluate(w, normalized);
}


// initialization of learning rate parameters
// run a number of different parameters on the normalized or unnormalized objective and choose
// those with best decrease
void StochasticGradientDescent::initializeLearningRate(Weights &w, double initialEta, int subsetSize, bool normalized) {

  const double factor = 10.0; //2.0;
  const double c      = 4.0;
  const int tries     = 10;
  
  double fval, initialFval, eta, lambda;  
  double bestFval, bestEta;
    
  SearchIx indices = objective->getNonEmpty();
  int numIndices = indices.size();

  cout << "Initializing parameters for learning rate..." << endl; 

  // reset epoch and and iteration t to zero
  epoch = 0;
  t = 0;

  // if alpha is set to zero, initialize to 1/(constant*num_training_examples)  
  lambda = objective->getLambda();
  if (alpha == 0.0) {
    alpha = 1/(c*numIndices); // because Leon says so!
  }
  
  // choose random subset of training set
  random_shuffle(indices.begin(), indices.end());

  // if no sample size is set, use entire training set
  SearchIx subset;
  if (subsetSize == 0 || subsetSize >= numIndices) {
    subset = indices;
  } else {
    subset = SearchIx(subsetSize);
    for (int k=0; k<subsetSize; k++) {
      subset[k] = indices[k];
    }
  }
  
  // compute initial fval
  initialFval = objective->evaluate(w, normalized);
  bestFval = tryEta(w, initialEta, subset, normalized);
  bestEta = eta = initialEta;
  
  if (isnan(bestFval)) {
    throw NOT_A_NUMBER;
  }
  
  cout << "initial eta = " << initialEta << ", initial fval = " << initialFval << endl;

  // run through different set of parameters and choose one that minimizes fval most
  // run minimum 10 tries
  Weights wNew(w.size()); 
  int i = 0;
  bool tryincrease = true;
  while (i < tries) {
      
    i++;
    cout << "Try " << i <<  " of minimum " << tries << endl;    

    // first try increasing eta
    if (tryincrease)     
      eta *= factor;
    else
      eta /= factor;
    
    // tryEta changes w, so use new vector
    wNew = w;

    fval = tryEta(wNew, eta, subset, normalized);
    cout << "testing: eta = " << eta << ", t0 = " << 1/(alpha*eta) << ", alpha = " << alpha << ", fval = " << fval << endl;
    if (!isnan(fval) && fval < initialFval) {  // must decrease original fval
      if (fval < bestFval) {
        bestFval = fval;
        bestEta = eta;
      }
    } else if (tryincrease) {
        eta = initialEta;
        tryincrease = false;
    }

  }
  
  // divide by factor to enforce implicit regularization (Bottou trick)
  //bestEta /= factor;
  this->t0 = 1.0/(alpha*bestEta);

  
  cout << "Initialization done!" << endl;  
  cout << "best parameters: eta = "  << bestEta << ", t0 = " << t0 << ", alpha = " << alpha << endl;
}


// implementation of SGD learning
Weights StochasticGradientDescent::learnWeights(const Weights &w) {

  int weightDim = w.size();
  Dvector grad(weightDim, 0.0);
  Weights wNew(w);
  Weights wAvg(weightDim, 0.0);

  // learning rate
  double eta;

  // get number of training examples with object
  SearchIx indices = objective->getNonEmpty();
  int numIndices = indices.size();
  
  // seed random number generator used in shuffling the dataset
  srand((unsigned)time(NULL)); rand();
  
  // main loop
  for (int j=0; j<maxEpochs; j++) {
    
    // update epoch number
    epoch++;

    // randomly shuffle dataset    
    random_shuffle(indices.begin(), indices.end());
    
    // print stuff
    // run through all training examples
    for (int i=0; i<numIndices; i++) {
      
      // update iteration number
      t++;  
      
      // compute gradient for one training example
      gradient->evaluate(grad, wNew, indices[i]);

      // update learning rate
      if (constLearningRate) {
        eta = min(1.0/(alpha*t0), 1.0);
      } else {
        eta = min(1.0/(alpha*(t+t0)), 1.0);
      }
      
      // update weights
      // w = w + eta*d = w - eta*gradient
      for (int i=0; i<weightDim; i++) {
        wNew[i] = wNew[i] - eta*grad[i];
      }
      
      // average weights from the last epoch
      if (epoch == maxEpochs) {
        for (int i=0; i<weightDim; i++) {
          wAvg[i] += wNew[i]/numIndices;
        }
      }   
    }
    
    // print progress and store temporary weights
    progress(wNew, wAvg, epoch, t, eta, epoch == maxEpochs);
    
  }

  // return averaged weights
  return wAvg;
}

// print progress and store temp weights
void StochasticGradientDescent::progress(Weights &w, Weights &wAvg, int epoch, int t, double eta, bool lastEpoch) {

  // compute current value of objective
  double fval = objective->evaluate(w);
  cout << "Epoch:      " << epoch << endl;
  cout << "Iterations: " << t << endl;
  cout << "eta:        " << eta << endl;
  cout << "fval:       " << fval << endl;
  
  if (lastEpoch) {
    double fvalAvg = objective->evaluate(wAvg);
    cout << "fvalAvg:    " << fvalAvg << endl;
  }
  
  cout << endl;
  
  // store weights
  ofstream tempWeightFile(tempWeightsPath.c_str());
  if (!tempWeightFile) {
    cerr << "Could not open file " << tempWeightsPath << endl;
  }
  
  // store result in file
  for (size_t i=0; i<w.size(); i++) {
    tempWeightFile << w[i] << "\n";
  }
  
  tempWeightFile.close();

}


