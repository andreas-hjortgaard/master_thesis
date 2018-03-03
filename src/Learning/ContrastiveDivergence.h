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

#ifndef _CONTRASTIVE_DIVERGENCE_H_
#define _CONTRASTIVE_DIVERGENCE_H_

#include "ObjectiveFunctions/SampledGradient.h"
#include "StochasticGradientDescent.h"

// contrastive divergence inherits its online features from stochastic gradient
class ContrastiveDivergence : public StochasticGradientDescent {
  
  private:

    SampledGradient *gradient;    

    // number of steps to take in the Gibbs chain
    int chainSteps;
    int numSamples;

  public:

    // constructors
    ContrastiveDivergence(ObjectiveFunction *obj, SampledGradient *grad);
    ContrastiveDivergence(ObjectiveFunction *obj, SampledGradient *grad, int k, double alpha_, double t0_);
    
};

#endif // _CONTRASTIVE_DIVERGENCE_H_
