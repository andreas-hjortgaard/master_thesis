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

#include <fstream>
#include "GradientDescent.h"

using namespace std;

// constructors
GradientDescent::GradientDescent(ObjectiveFunction *obj, Gradient *grad) : objective(obj), gradient(grad) { }

// getters/setters
ObjectiveFunction *GradientDescent::getObjective() {
  return objective;
}

void GradientDescent::setObjective(ObjectiveFunction *obj) {
  objective = obj;
}

Gradient *GradientDescent::getGradient() {
  return gradient;
}

void GradientDescent::setGradient(Gradient *grad) {
  gradient = grad;
}

