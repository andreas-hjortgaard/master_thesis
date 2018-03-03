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

// implementation of piecewise log-likelihood objective function
#include <cmath>
#include <iostream>

#include "PiecewiseLogLikelihood.h"

using namespace std;

// constructor
PiecewiseLogLikelihood::PiecewiseLogLikelihood(DataManager *dm, PiecewiseConditionalRandomField *crfield, SearchIx si)
  : ObjectiveFunction::ObjectiveFunction(dm, crfield, si), crf(crfield)
{
}

// evaluate (specific to the actual objective function)
double PiecewiseLogLikelihood::evaluate(Weights &w, bool normalized) {
  
  int imageNumber;
  Bboxes &bboxes = dataManager->getBboxes();
  IntegralImage *integralImage;
  int stepSize = getStepSize();
  
  // compute regularizer
  double regularizer = 0.0;
  for (size_t i = 0; i < w.size(); i++) {
    regularizer += w[i]*w[i];
  }
  regularizer *= lambda;
  
  // compute dot product with feature map
  double dotproduct = 0.0;
  int xl,yl,xh,yh;

  // compute log Z (normalizing constant)
  double logZ = 0.0;
  Dvector logZ_F (4);

  for (size_t i=0; i<searchIx.size(); i++) {

    // extract image index
    imageNumber = searchIx[i];

    // only compute integral image when needed
    if (bboxes[imageNumber].numObject > 0) {
    
      // compute integral image
      computeIntegralImage(imageNumber, w);   
      integralImage = crf->getIntegralImage();
  
      // compute log Z_F
      crf->slidingWindowLogSumExp(logZ_F);

      // only work on images that actually contain the object
      for (int numObj = 0; numObj < bboxes[imageNumber].numObject; numObj++) {
        
        // calculate score on ground truth bounding box
        // fit to quantized integralImage space
        xl = min(bboxes[imageNumber].ltrb[4*numObj+0]/stepSize, iiWidth-2);
        yl = min(bboxes[imageNumber].ltrb[4*numObj+1]/stepSize, iiHeight-2);
        xh = min(bboxes[imageNumber].ltrb[4*numObj+2]/stepSize, iiWidth-2);
        yh = min(bboxes[imageNumber].ltrb[4*numObj+3]/stepSize, iiHeight-2);
        
        dotproduct += (*integralImage)[crf->iiOffset(xl,yl)];
        dotproduct -= (*integralImage)[crf->iiOffset(xl,yh+1)];
        dotproduct -= (*integralImage)[crf->iiOffset(xh+1,yl)];
        dotproduct += (*integralImage)[crf->iiOffset(xh+1,yh+1)];
        
        logZ += logZ_F[0];
        logZ += logZ_F[1];
        logZ += logZ_F[2];
        logZ += logZ_F[3];

      }
    }
  }
  
  double loglik = regularizer - dotproduct + logZ;

  //printf("regularizer = %.6f\n", regularizer);
  //printf("dotproduct  = %.6f\n", dotproduct);
  //printf("logZ        = %.6f\n", logZ);
  //printf("loglik      = %.6f\n\n", loglik);
  
  if (isnan(loglik)) {
    throw NOT_A_NUMBER;
  }
  return loglik;
}


