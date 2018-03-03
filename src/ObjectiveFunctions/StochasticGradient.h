#ifndef _STOCHASTIC_GRADIENT_H_
#define _STOCHASTIC_GRADIENT_H_

#include "LogLikelihoodGradient.h"

// stochastic gradient derived from the loglikelihood gradient class
class StochasticGradient : public LogLikelihoodGradient {

  public:
  
    // constructor
    StochasticGradient(DataManager *dm=NULL, ConditionalRandomField *crfield=NULL, SearchIx si=SearchIx());

    // evaluate stochastic
    virtual void evaluate(Dvector &gradient, Weights &w, int imageNumber, bool normalized = true);

};

#endif // _STOCHASTIC_GRADIENT_H_

