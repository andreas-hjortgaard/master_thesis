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

#include "LogLikelihood.h"

using namespace std;

// constructor
LogLikelihood::LogLikelihood(DataManager *dm, ConditionalRandomField *crfield, SearchIx si)
  : ObjectiveFunction::ObjectiveFunction(dm, crfield, si)
{
}

// evaluate (specific to the actual objective function)
double LogLikelihood::evaluate(Weights &w, bool normalized) {

  int imageNumber;
  double regularizer, dotproduct, logZ, currentLogZ;
  
  int weightDim = w.size();
  int stepSize = getStepSize(); // retrieve step size from CRF

  Bboxes &bboxes = dataManager->getBboxes();

  // compute regularizer
  regularizer = 0.0;
  for (int i=0; i<weightDim; i++) {
    regularizer += w[i]*w[i];
  }
  regularizer *= lambda;

  // compute dot product with feature map
  // compute log Z (normalizing constant)
  dotproduct = 0.0;
  logZ = 0.0; 
  for (size_t i=0; i<searchIx.size(); i++) {

    // extract image index
    imageNumber = searchIx[i];

    // only compute integral image when needed
    if (bboxes[imageNumber].numObject > 0) {
      // compute integral image
      computeIntegralImage(imageNumber, w);

      // compute logZ for current image 
      if (normalized) {
        currentLogZ = slidingWindowLogSumExp();
      }
    }

    // only work on images that actually contain the object
    for (int numObj = 0; numObj < bboxes[imageNumber].numObject; numObj++) {

      // calculate score on ground truth bounding box
      // fit to quantized integralImage space
      // calls computeBboxScore(left, top, right, bottom) quantized
      dotproduct += computeBboxScore(min(bboxes[imageNumber].ltrb[4*numObj+0]/stepSize, iiWidth-2),
                                     min(bboxes[imageNumber].ltrb[4*numObj+1]/stepSize, iiHeight-2),
                                     min(bboxes[imageNumber].ltrb[4*numObj+2]/stepSize, iiWidth-2),
                                     min(bboxes[imageNumber].ltrb[4*numObj+3]/stepSize, iiHeight-2));
      if (normalized) {
        logZ += currentLogZ;
      }
    }
  }

  double loglik = regularizer - dotproduct + logZ;

  //cout << "regularizer = " << regularizer << endl;
  //cout << "dotproduct  = " << dotproduct << endl;
  //cout << "logZ        = " << logZ << endl;
  //cout << "loglik      = " << loglik << endl << endl;
  
  //if (isnan(loglik)) {
  //  throw NOT_A_NUMBER;
  //}
  return loglik;
}

