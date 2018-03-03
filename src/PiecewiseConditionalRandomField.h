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

#ifndef _PIECEWISE_CONDITIONAL_RANDOM_FIELD_H_
#define _PIECEWISE_CONDITIONAL_RANDOM_FIELD_H_

#include "DataManager.h"
#include "ConditionalRandomField.h"


// Piecewise Conditional Random Field class
// for computing probabilities of bounding boxes
// as well as conditional probabilities of the individual box coordinates
class PiecewiseConditionalRandomField : public ConditionalRandomField {
  
  public:
    
    PiecewiseConditionalRandomField(DataManager *dataman);  
    
    // only difference is the normalization
    void slidingWindowLogSumExp(Dvector &logZ_F); 
    
    // marginal probability of one corner (that is two connected sides) of the bbox
    double cornerP(int xvar, int yvar, const Bbox &bbox, int imageNumber, const Weights &w, bool computeIIlogZ = true, Dvector logZ_F = Dvector(4,0.0), double maxScore = 0.0);

};

#endif // _PIECEWISE_CONDITIONAL_RANDOM_FIELD_H_
