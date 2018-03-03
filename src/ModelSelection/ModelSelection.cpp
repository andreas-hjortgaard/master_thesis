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

// functions for doing model selection using either a validation set or by using
// cross validation

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>

#include "DataManager.h"
#include "ConditionalRandomField.h"
#include "Measures/LossMeasures.h"
#include "ObjectiveFunctions/ObjectiveFunction.h"
#include "ObjectiveFunctions/Gradient.h"
#include "ObjectiveFunctions/StochasticGradient.h"
#include "Learning/GradientDescent.h"
#include "Learning/StochasticGradientDescent.h"
#include "Types.h"

using namespace std;


RecallOverlap modelSelection(Weights &wNew,
                      double lambda,
                      Weights &initial,
                      DataManager &trainingSet, 
                      DataManager &validationSet, 
                      ConditionalRandomField &crf, 
                      ObjectiveFunction &objective,
                      Gradient &gradient, 
                      GradientDescent &learningAlg) {
  
  // set lambda
  objective.setLambda(lambda);
  gradient.setLambda(lambda);

  cout << "Trying lambda = " << lambda << endl;

  wNew = learningAlg.learnWeights(initial);
  validationSet.setWeights(wNew);

  // check loss on validation set
  SearchIx indices;
  RecallOverlap recallOverlap;
  recallOverlap = computeRecallOverlap(validationSet, indices, crf.getStepSize());

  return recallOverlap;

}

// perform model selection on validation set
// min and max determines the power of 2 to be used such that the range of
// parameter values is [2^min, 2^max]
double modelSelection(DataManager &trainingSet, 
                      DataManager &validationSet, 
                      ConditionalRandomField &crf, 
                      ObjectiveFunction &objective,
                      Gradient &gradient, 
                      GradientDescent &learningAlg, 
                      int min, 
                      int max) {

  // outline:
  // - train with lambda = 2^min on training set
  // - check loss on validation set
  // - repeat until lambda = 2^max
  
  double lambda, bestLambda;
  double bestRecallOverlap = -9999999999.;
  
  int weightDim = crf.getWeightDim();
  Weights w(weightDim, 0.1);
  Weights wNew(weightDim, 0.0);
  
  RecallOverlap recallOverlap;

  // test different lambda values
  for (int p=min; p<=max; p++) {
    
    lambda = pow((double)10, p);    
    
    recallOverlap = modelSelection(wNew, lambda, w, trainingSet, validationSet, crf, objective, gradient, learningAlg);
    
    // update best lambda
    if (recallOverlap.AUC > bestRecallOverlap) {
      bestRecallOverlap = recallOverlap.AUC;
      bestLambda = lambda;
    }
  }

  return bestLambda;
}


// perform model selection on validation set
// min and max determines the power of 2 to be used such that the range of
// parameter values is [2^min, 2^max]
double modelSelectionStochastic(DataManager &trainingSet, 
                      DataManager &validationSet, 
                      ConditionalRandomField &crf, 
                      ObjectiveFunction &objective,
                      StochasticGradient &gradient, 
                      StochasticGradientDescent &learningAlg, 
                      int min, 
                      int max) {

  // outline:
  // - train with lambda = 2^min on training set
  // - check loss on validation set
  // - repeat until lambda = 2^max
  
  double lambda, bestLambda;
  double bestRecallOverlap = -9999999999.;
  
  int weightDim = crf.getWeightDim();
  Weights w(weightDim, 0.1);
  Weights wNew(weightDim, 0.0);

  // test different lambda values
  for (int p=min; p<=max; p++) {
    // set lambda
    lambda = pow((double)10, p);
    objective.setLambda(lambda);
    gradient.setLambda(lambda);

    cout << "Trying lambda = " << lambda << endl;

    learningAlg.initializeLearningRate(w, 0.1, 0, false);
    wNew = learningAlg.learnWeights(w);
    //crf.setWeights(wNew);    
    validationSet.setWeights(wNew);

    // check loss on validation set
    SearchIx indices;
    RecallOverlap recallOverlap;
    recallOverlap = computeRecallOverlap(validationSet, indices, crf.getStepSize());

    // save learned weights in file
    ostringstream os;
    os << "lambda_" << lambda << ".txt";
    ofstream weightFile(os.str().c_str());
    if (!weightFile) {
      cerr << "Could not open file..." << endl;
    }
    
    // store result in file
    for (int i=0; i<weightDim; i++) {
      weightFile << wNew[i] << endl;
    }
    
    weightFile.close();
    
    // update best lambda
    if (recallOverlap.AUC > bestRecallOverlap) {
      bestRecallOverlap = recallOverlap.AUC;
      bestLambda = lambda;
    }
  }

  return bestLambda;
}

// perform model selection by cross-validation
// min and max determines the power of 2 to be used such that the range of
// parameter values is [2^min, 2^max]
/*double crossValidation(DataManager &trainingSet, 
                      DataManager &validationSet, 
                      ConditionalRandomField &crf, 
                      ObjectiveFunction &objective,
                      Gradient &gradient, 
                      GradientDescent &learningAlg, 
                      int folds,
                      int min, 
                      int max) {

  // outline:
  // - divide training set into k equally large subsets
  // - train with lambda = 2^min on k-1 subsets
  // - compute loss on the k'th subset
  // - repeat for all subsets
  // - compute average loss
  // - repeat until lambda = 2^max

  int numImages, numSubsets;
  Images train;
  SearchIx subsetTrain, subsetVal;

  // set up objective functions
  PseudoLikelihood pseudolik(trainingSet, crf);
  PseudoLikelihoodGradient pseudolikgrad(trainingSet, crf);
  LBFGS lbfgs(pseudolik, pseudolikgrad);

  // retrieve images
  numImages = trainingSet.getNumImages();
  train = trainingSet.getImages();
  
  // divide training set into k subsets    
  srand(time(NULL)); // seed for randomizer  
  random_shuffle(train.begin(), train.end());
  
  for (int i=0; i<k; i++) {
    
    // create subsets
    subsetTrain = 

    // train model

    // compute loss

    // accumulate average loss
  
  }
  


  return 2.0;
}*/
