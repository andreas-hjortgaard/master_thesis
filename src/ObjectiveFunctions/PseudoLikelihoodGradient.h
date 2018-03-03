#ifndef _PSEUDO_LIKELIHOOD_GRADIENT_H_
#define _PSEUDO_LIKELIHOOD_GRADIENT_H_

#include "Gradient.h"

// Pseudo-likelihood gradient derived from the gradient class
class PseudoLikelihoodGradient : public Gradient {

  private:
    
    // sliding window for expectation computation
    void slidingWindowExpectation(Dvector &expectation, Ivector &featureMap, int imageNumber, Bbox &scaledBbox, int s);
  

  public:
  
    // constructor
    PseudoLikelihoodGradient(DataManager *dm=NULL, ConditionalRandomField *crfield=NULL, SearchIx si=SearchIx());

    // evaluate (specific to the actual objective function)
    virtual void evaluate(Dvector &gradient, Weights &w, bool normalized = true);    

};

#endif // _PSEUDO_LIKELIHOOD_GRADIENT_H_
