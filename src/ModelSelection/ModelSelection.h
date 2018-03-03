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

#include "DataManager.h"
#include "ConditionalRandomField.h"
#include "Types.h"
#include "ObjectiveFunctions/ObjectiveFunction.h"
#include "ObjectiveFunctions/Gradient.h"
#include "ObjectiveFunctions/StochasticGradient.h"
#include "Learning/GradientDescent.h"
#include "Learning/StochasticGradientDescent.h"


// perform model selection on validation set
// min and max determines the power of 2 to be used such that the range of
// parameter values is [2^min, 2^max]
RecallOverlap modelSelection(Weights &wNew,
                      double lambda,
                      Weights &initial,                      
                      DataManager &trainingSet, 
                      DataManager &validationSet, 
                      ConditionalRandomField &crf, 
                      ObjectiveFunction &objective,
                      Gradient &gradient, 
                      GradientDescent &learningAlg);

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
                      int max);

// perform model selection with stochastic gradient or contrastive divergence
double modelSelectionStochastic(DataManager &trainingSet, 
                      DataManager &validationSet, 
                      ConditionalRandomField &crf, 
                      ObjectiveFunction &objective,
                      StochasticGradient &gradient, 
                      StochasticGradientDescent &learningAlg, 
                      int min, 
                      int max);

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
                      int max);*/
