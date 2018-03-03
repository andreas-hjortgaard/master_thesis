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

// implementation of log-likelihood objective function
#include <cmath>
#include <iostream>

#include "LogLikelihoodGradient.h"

using namespace std;

// constructor
LogLikelihoodGradient::LogLikelihoodGradient(DataManager *dm, ConditionalRandomField *crfield, SearchIx si)
  : Gradient::Gradient(dm, crfield, si)
{
}

// gradient (specific to the actual objective function)
void LogLikelihoodGradient::evaluate(Dvector &gradient, Weights &w, bool normalized) {

  int imageNumber;
  Bboxes &bboxes = dataManager->getBboxes();
  int stepSize = getStepSize();
  int weightDim = w.size();

  // check gradient size
  if (gradient.size() != w.size()) {
    throw GRADIENT_SIZE_ERROR;
  }
  
  // create temporary Dvector
  Ivector featureMap      = Ivector(weightDim, 0);
  Ivector tempFeatureMap  = Ivector(weightDim, 0);
  Dvector expectation     = Dvector(weightDim, 0.0);
  
  // compute gradient regularizer
  // (no need to reset gradient before computation, just overwrite!)
  for (int i=0; i<weightDim; i++) {
    gradient[i] = 2*lambda*w[i];
  }

  // compute feature map and expectation
  for (size_t j=0; j<searchIx.size(); j++) {

    // extract image index
    imageNumber = searchIx[j];

    // only compute integral image and integral histogram when necessary
    if (bboxes[imageNumber].numObject > 0) {
      
      // compute integral image
      computeIntegralImage(imageNumber, w);
      
      // compute integral histogram
      computeIntegralHistogram(imageNumber);

      // compute expectation
      if (normalized) {
        slidingWindowExpectation(expectation, tempFeatureMap);
      }
    }

    // only work on images that actually contain the object
    for (int numObj = 0; numObj < bboxes[imageNumber].numObject; numObj++) {
    
      // compute feature map
      // fit to quantized integralImage space
      computeFeatureMap(featureMap,
                        min(bboxes[imageNumber].ltrb[4*numObj+0]/stepSize, iiWidth-2),
                        min(bboxes[imageNumber].ltrb[4*numObj+1]/stepSize, iiHeight-2),
                        min(bboxes[imageNumber].ltrb[4*numObj+2]/stepSize, iiWidth-2),
                        min(bboxes[imageNumber].ltrb[4*numObj+3]/stepSize, iiHeight-2));
      
      
      // update gradient
      for (int i=0; i<weightDim; i++) {
        gradient[i] -= featureMap[i] - expectation[i];
      }
    }
  }
}



// SLIDING WINDOW FUNCTIONS

// sliding window using expectation
void LogLikelihoodGradient::slidingWindowExpectation(Dvector &expectation, Ivector &featureMap)
{
  double logZ, p;
  int weightDim = expectation.size();

  // clear and reset expectation
  expectation.clear();
  expectation.resize(weightDim, 0.0);

  // compute normalization constant
  logZ = slidingWindowLogSumExp();
  
  // compute p(l,t,r,b) = p(l,r|x)p(t,b|x)
  for (short bbox_h = 0; bbox_h < iiHeight - 1; bbox_h++) {
    //for all bounding widths
    for (short bbox_w = 0; bbox_w < iiWidth - 1; bbox_w++) {
      //for all Top-Left y-coordinates
      for (short y = 0; y < iiHeight - bbox_h - 1; y++) {
        //all Top-Left x-coordinates
        for (short x = 0; x < iiWidth - bbox_w - 1; x++) {
          p = exp(computeBboxScore(x, y, x+bbox_w, y+bbox_h) - logZ);
          computeFeatureMap(featureMap, x, y, (x+bbox_w), (y+bbox_h));
          for (int i=0; i<weightDim; i++) {
            expectation[i] += p*featureMap[i];
          }
        }
      }
    }
  }
}

