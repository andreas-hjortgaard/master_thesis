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

#ifndef _PIECEWISE_LOG_LIKELIHOOD_H_
#define _PIECEWISE_LOG_LIKELIHOOD_H_

#include "ObjectiveFunction.h"
#include "PiecewiseConditionalRandomField.h"

// piecewise log-likelihood derived from the objective function class
class PiecewiseLogLikelihood : public ObjectiveFunction {

  private:
    
    PiecewiseConditionalRandomField *crf;

  public:
  
    // constructor
    PiecewiseLogLikelihood(DataManager *dm=NULL, PiecewiseConditionalRandomField *crf=NULL, SearchIx si=SearchIx());

    // evaluate (specific to the actual objective function)
    virtual double evaluate(Weights &w, bool normalized = true);

};

#endif // _PIECEWISE_LOG_LIKELIHOOD_H_

