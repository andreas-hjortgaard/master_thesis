#ifndef _PSEUDO_LIKELIHOOD_H_
#define _PSEUDO_LIKELIHOOD_H_

#include "ObjectiveFunction.h"

// Pseudo-likelihood derived from the objective function class
class PseudoLikelihood : public ObjectiveFunction {

  public:
  
    // constructor
    PseudoLikelihood(DataManager *dm=NULL, ConditionalRandomField *crfield=NULL, SearchIx si=SearchIx());

    // evaluate pseudolikelihood
    virtual double evaluate(Weights &w, bool normalized = true);    

};

#endif // _PSEUDO_LIKELIHOOD_H_

