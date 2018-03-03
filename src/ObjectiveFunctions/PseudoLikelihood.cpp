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

// implementation of pseudo-likelihood objective function
#include <cmath>

#include "PseudoLikelihood.h"

using namespace std;

// constructor
PseudoLikelihood::PseudoLikelihood(DataManager *dm, ConditionalRandomField *crfield, SearchIx si)
  : ObjectiveFunction::ObjectiveFunction(dm, crfield, si)
{
}

// evaluate pseudolikelihood
double PseudoLikelihood::evaluate(Weights &w, bool normalized) {

  int imageNumber;
  double regularizer, dotproduct, logZs;

  int weightDim = w.size();
  Bboxes &bboxes = dataManager->getBboxes();
  int stepSize = getStepSize();  

  // compute regularizer
  regularizer = 0.0;
  for (int i=0; i<weightDim; i++) {
    regularizer += w[i]*w[i];
  }
  regularizer *= lambda;

  // compute dot product with feature map
  // compute log Z (normalizing constant)
  dotproduct = 0.0;
  logZs = 0.0;
  
  Bbox scaledBbox;
  scaledBbox.ltrb = new short[4];
  
  //for (int n=0; n<num_files; n++) {
  for (size_t i=0; i<searchIx.size(); i++) {
    
    // extract image index
    imageNumber = searchIx[i];
    
    // only compute integral image when necessary.
    if (bboxes[imageNumber].numObject > 0) {
      // compute integral image
      computeIntegralImage(imageNumber, w);
    }

    // only work on images that actually contain the object      
    for (int numObj = 0; numObj < bboxes[imageNumber].numObject; numObj++) {
    
      // scale true bbox
      scaledBbox.ltrb[LEFT]   = min(bboxes[imageNumber].ltrb[4*numObj+0]/stepSize, iiWidth-2);
      scaledBbox.ltrb[TOP]    = min(bboxes[imageNumber].ltrb[4*numObj+1]/stepSize, iiHeight-2);
      scaledBbox.ltrb[RIGHT]  = min(bboxes[imageNumber].ltrb[4*numObj+2]/stepSize, iiWidth-2);
      scaledBbox.ltrb[BOTTOM] = min(bboxes[imageNumber].ltrb[4*numObj+3]/stepSize, iiHeight-2);

      // calculate score on ground truth bounding box  
      dotproduct += computeBboxScore(scaledBbox.ltrb[LEFT],
                                     scaledBbox.ltrb[TOP],
                                     scaledBbox.ltrb[RIGHT],
                                     scaledBbox.ltrb[BOTTOM]);
      
      // Vary one of (left, top, right, bottom), keep all other constant
      for (int s=0; s<4; s++) {
        logZs += crf->slidingWindowLogSumExpCond(s, scaledBbox);
      }
    }
  }
  
  delete[] scaledBbox.ltrb;
  
  //printf("\nregularizer = %.6f\n", regularizer);
  //printf("dotproduct = %.6f\n", dotproduct);
  //printf("logZ = %.6f\n", logZs);
  //printf("pseudo-loglik = %.6f\n", regularizer - 4*dotproduct + logZs);
  return regularizer - 4*dotproduct + logZs;
}

