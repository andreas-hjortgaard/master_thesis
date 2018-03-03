// implementation of pseudo-likelihood gradient
#include <cstdio>
#include <cmath>

#include "PseudoLikelihoodGradient.h"

using namespace std;

// constructor
PseudoLikelihoodGradient::PseudoLikelihoodGradient(DataManager *dm, ConditionalRandomField *crfield, SearchIx si)
  : Gradient::Gradient(dm, crfield, si)
{
}

// gradient (specific to the actual objective function)
void PseudoLikelihoodGradient::evaluate(Dvector &gradient, Weights &w, bool normalized) {

  int imageNumber;
  Bboxes &bboxes = dataManager->getBboxes();
  int stepSize = getStepSize();
  int weightDim = w.size(); 

  // check gradient size
  if (gradient.size() != w.size()) {
    throw GRADIENT_SIZE_ERROR;
  }

  Bbox scaledBbox;
  scaledBbox.ltrb = new short[4];

  // create temporary Dvector
  Ivector featureMap      = Ivector(w.size(), 0);
  Ivector tempFeatureMap  = Ivector(w.size(), 0);
  Dvector expectation     = Dvector(w.size(), 0.0);
  
  // compute gradient regularizer
  // (no need to reset gradient before computation, just overwrite!)
  for (int i=0; i<weightDim; i++) {
    gradient[i] = 2*lambda*w[i];
  }

  // compute feature map and expectation
  for (size_t j=0; j<searchIx.size(); j++) {

    // extract image index
    imageNumber = searchIx[j];
    
    if (bboxes[imageNumber].numObject > 0) {
  
      // compute integral image
      computeIntegralImage(imageNumber, w);
      
      // compute integral histogram
      computeIntegralHistogram(imageNumber);
    }

    // only work on images that actually contain the object
    for (int numObj = 0; numObj < bboxes[imageNumber].numObject; numObj++) {
    
      // scale true bbox
      scaledBbox.ltrb[LEFT]   = min(bboxes[imageNumber].ltrb[4*numObj+0]/stepSize, iiWidth-2);
      scaledBbox.ltrb[TOP]    = min(bboxes[imageNumber].ltrb[4*numObj+1]/stepSize, iiHeight-2);
      scaledBbox.ltrb[RIGHT]  = min(bboxes[imageNumber].ltrb[4*numObj+2]/stepSize, iiWidth-2);
      scaledBbox.ltrb[BOTTOM] = min(bboxes[imageNumber].ltrb[4*numObj+3]/stepSize, iiHeight-2);

      // compute feature map
      // divide and multiply by stepSize to discritize same way as integralImage is discretized
      computeFeatureMap(featureMap, 
                        scaledBbox.ltrb[LEFT],
                        scaledBbox.ltrb[TOP],
                        scaledBbox.ltrb[RIGHT],
                        scaledBbox.ltrb[BOTTOM]);
      
      // update gradient
      for (int i=0; i<weightDim; i++) {
        gradient[i] -= 4*featureMap[i];
      }
      
      // Vary one of (left, top, right, bottom), keep all other constant
      for (int s=0; s<4; s++) {
  
        // compute expectation
        slidingWindowExpectation(expectation, tempFeatureMap, imageNumber, scaledBbox, s);
        
        // update gradient
        for (int i=0; i<weightDim; i++) {
          gradient[i] += expectation[i];
        }
      }
    }
  }
  
  delete[] scaledBbox.ltrb;
}


// SLIDING WINDOW FUNCTIONS

// sliding window using expectation
void PseudoLikelihoodGradient::slidingWindowExpectation(Dvector &expectation, Ivector &featureMap, int imageNumber, Bbox &scaledBbox, int s)
{
  double logZs, p;
  int weightDim = expectation.size();

  // clear and reset expectation
  expectation.clear();
  expectation.resize(weightDim, 0.0);
  
  // select the boundaries for the given variable
  short start, stop;
  switch (s) {
    case LEFT:
      // left goes from left edge to y_r
      start = 0;
      stop = scaledBbox.ltrb[RIGHT];
      break;
    case TOP:
      // top goes from top edge to y_b
      start = 0;
      stop = scaledBbox.ltrb[BOTTOM];
      break;
    case RIGHT:
      // right goes from y_r to right edge
      start = scaledBbox.ltrb[LEFT];
      stop  = iiWidth-2;
      break;
    case BOTTOM:
      // bottom goes from y_t to bottom edge
      start = scaledBbox.ltrb[TOP];
      stop  = iiHeight-2;
      break;
  }
  
  // store original value
  int val = scaledBbox.ltrb[s];

  // compute normalization constant from CRF
  logZs = crf->slidingWindowLogSumExpCond(s, scaledBbox);
  
  // compute expectation
  for (short i = start; i <= stop; i++) {
    scaledBbox.ltrb[s] = i;
    p = exp(computeBboxScore(scaledBbox.ltrb[LEFT], scaledBbox.ltrb[TOP], scaledBbox.ltrb[RIGHT], scaledBbox.ltrb[BOTTOM]) - logZs);
    computeFeatureMap(featureMap, scaledBbox.ltrb[LEFT], scaledBbox.ltrb[TOP], scaledBbox.ltrb[RIGHT], scaledBbox.ltrb[BOTTOM]);
    for (int j=0; j<weightDim; j++) {
      expectation[j] += p*featureMap[j];
    }
  }
  
  // reinsert original value  
  scaledBbox.ltrb[s] = val;

}

