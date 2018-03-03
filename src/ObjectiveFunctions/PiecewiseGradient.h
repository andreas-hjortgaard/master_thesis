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

#ifndef _PIECEWISE_GRADIENT_H_
#define _PIECEWISE_GRADIENT_H_

#include "Gradient.h"
#include "PiecewiseConditionalRandomField.h"

// piecewise likelihood gradient derived from the gradient class
class PiecewiseGradient : public Gradient {

  private:

    // uses piecewise conditional random field
    PiecewiseConditionalRandomField *crf;

    // sliding window for expectation computation
    void slidingWindowExpectation(Dvector &expectation, int imageNumber);

    // convert (x,y) into 1d index (from CRF)
    int iiOffset(int x, int y);

  public:
  
    // constructor
    PiecewiseGradient(DataManager *dm=NULL, PiecewiseConditionalRandomField *crfield=NULL, SearchIx si=SearchIx());

    // evaluate (specific to the actual objective function)
    virtual void evaluate(Dvector &gradient, Weights &w, bool normalized = true);    

};

#endif // _PIECEWISE_GRADIENT_H_
