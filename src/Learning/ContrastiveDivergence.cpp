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
#include <cmath>
#include <cstdlib>
#include <algorithm>

#include "Types.h"
#include "ContrastiveDivergence.h"

using namespace std;


// implementation of the contrastive divergence

// constructors (initialize CD parameters)
ContrastiveDivergence::ContrastiveDivergence(ObjectiveFunction *obj, SampledGradient *grad) : 
  StochasticGradientDescent(obj, grad), 
  gradient(grad),  
  chainSteps(1),
  numSamples(1) 
{ 
  gradient->setNumSamples(numSamples);
  gradient->setSkip(chainSteps);
}

ContrastiveDivergence::ContrastiveDivergence(ObjectiveFunction *obj, SampledGradient *grad, int k, double alpha_, double t0_) : 
  StochasticGradientDescent(obj, grad, alpha_, t0_), 
  gradient(grad),
  chainSteps(k),
  numSamples(1) 
{ 
  gradient->setNumSamples(numSamples);
  gradient->setSkip(chainSteps);
}

