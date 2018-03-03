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

#ifndef _GRADIENT_DESCENT_H_
#define _GRADIENT_DESCENT_H_

#include "ObjectiveFunctions/ObjectiveFunction.h"
#include "ObjectiveFunctions/Gradient.h"

// gradient descent learning
// abstract base class for all learning algorithms
class GradientDescent {
  
  protected:

    // objective to optimize        
    ObjectiveFunction *objective;
    
    // gradient
    Gradient *gradient;

  public: 

    // constructor
    GradientDescent(ObjectiveFunction *obj, Gradient *grad);
  
    // getters/setters
    ObjectiveFunction *getObjective();
    void setObjective(ObjectiveFunction *obj);

    Gradient *getGradient();
    void setGradient(Gradient *grad);

    // main function (takes a starting point as input)
    // is virtual so that the most derived version is used
    virtual Weights learnWeights(const Weights &w) = 0;

};


#endif // _GRADIENT_DESCENT_H_
