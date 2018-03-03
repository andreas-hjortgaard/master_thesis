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

#ifndef _GIBBS_SAMPLER_H_
#define _GIBBS_SAMPLER_H_

#include "ConditionalRandomField.h"
#include "Types.h"

// Gibbs sampler for sampling bounding boxes given a specific distribution
class GibbsSampler {

  private:
    
    ConditionalRandomField *crf;
  
    Dvector cumulativeHistogram;
    int histogramOffset;
    Ivector step;         // the current step in Gibbs chain
    Bboxes current;       // current state in the Gibbs chain (Bbox can handle several boxes at the same time)
                          // have one for each image

    // compute cumulative histogram
    void computeCumulativeHistogram(int var, Weights &w, int imageNumber);

    // sample one variable
    void sampleOne(int var, Weights &w, int imageNumber);
  
  public:

    // constructor
    GibbsSampler(ConditionalRandomField *crf);
    GibbsSampler(ConditionalRandomField *crf, Bboxes initial);
  
    // deconstructor
    ~GibbsSampler();
   
    // initialize Gibbs chain with random bboxes
    void initialize();

    // initialize Gibbs chain for the given image
    void initializeCurrent(int imageNumber, Bbox &bbox);

    // return current state of the Gibbs chain
    Bboxes *getCurrentSample();
  
    // return current state of the Gibbs chain for a single image
    Bbox *getCurrentSample(int imageNumber);

    // return cumulative histogram
    Dvector getCumulativeHistogram();

    // take k steps of the Gibbs chain to obtain one sample
    void sample(int k, Weights &w, int imageNumber, bool computeII = true);

};

#endif // _GIBBS_SAMPLER_H_
