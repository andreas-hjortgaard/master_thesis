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

#ifndef _LOG_LIKELIHOOD_GRADIENT_MPI_H_
#define _LOG_LIKELIHOOD_GRADIENT_MPI_H_

#include "Gradient.h"
#include "mpi.h"

// log-likelihood gradient derived from the gradient class
class LogLikelihoodGradient_MPI : public Gradient {

  private:
    
    // computing the expectation over bounding boxes using sliding windows
    void slidingWindowExpectation(Dvector &expectation, Ivector &featureMap);


  public:
  
    // constructor
    LogLikelihoodGradient_MPI(DataManager *dm=NULL, ConditionalRandomField *crfield=NULL, SearchIx si=SearchIx());

    // evaluate (specific to the actual gradient)
    virtual void evaluate(Dvector &gradient, Weights &w, bool normalized = true);

};

#endif // _LOG_LIKELIHOOD_GRADIENT_MPI_H_

