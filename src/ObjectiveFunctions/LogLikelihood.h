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

#ifndef _LOG_LIKELIHOOD_H_
#define _LOG_LIKELIHOOD_H_

#include "ObjectiveFunction.h"

// log-likelihood derived from the objective function class
class LogLikelihood : public ObjectiveFunction {

  public:
  
    // constructor
    LogLikelihood(DataManager *dm=NULL, ConditionalRandomField *crfield=NULL, SearchIx si=SearchIx());

    // evaluate log-likelihood
    virtual double evaluate(Weights &w, bool normalized = true);

};

#endif // _LOG_LIKELIHOOD_H_

