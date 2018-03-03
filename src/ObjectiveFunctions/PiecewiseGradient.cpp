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

// implementation of piecewise loglikelihood gradient
#include <cmath>

#include "PiecewiseGradient.h"

using namespace std;

// constructor
PiecewiseGradient::PiecewiseGradient(DataManager *dm, PiecewiseConditionalRandomField *crfield, SearchIx si)
  : Gradient::Gradient(dm, crfield, si), crf(crfield)
{
}

// convert (x,y) into 1d index
inline int PiecewiseGradient::iiOffset(int x, int y) {
  return crf->iiOffset(x,y);
}


// gradient (specific to the actual objective function)
void PiecewiseGradient::evaluate(Dvector &gradient, Weights &w, bool normalized) {

  int imageNumber;
  Bboxes &bboxes = dataManager->getBboxes();
  IntegralHistogram *integralHistogram;
  int stepSize = getStepSize();
  int weightDim = w.size(); 

  int xl,yl,xh,yh;
  
  // check gradient size
  if (gradient.size() != w.size()) {
    throw GRADIENT_SIZE_ERROR;
  }

  // create temporary Dvector
  Dvector expectation = Dvector(weightDim, 0.0);
  
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
      integralHistogram = crf->getIntegralHistogram();
           
      // compute expectation
      slidingWindowExpectation(expectation, imageNumber);
      
      // only work on images that actually contain the object
      for (int numObj = 0; numObj < bboxes[imageNumber].numObject; numObj++) {
        
        // fit to quantized integralImage space
        xl = min(bboxes[imageNumber].ltrb[4*numObj+0]/stepSize, iiWidth-2);
        yl = min(bboxes[imageNumber].ltrb[4*numObj+1]/stepSize, iiHeight-2);
        xh = min(bboxes[imageNumber].ltrb[4*numObj+2]/stepSize, iiWidth-2);
        yh = min(bboxes[imageNumber].ltrb[4*numObj+3]/stepSize, iiHeight-2);
        
        // compute feature map
        Ivector &featureMap_xlyl = (*integralHistogram)[iiOffset(xl,yl)];
        Ivector &featureMap_xlyh = (*integralHistogram)[iiOffset(xl,yh+1)];
        Ivector &featureMap_xhyl = (*integralHistogram)[iiOffset(xh+1,yl)];
        Ivector &featureMap_xhyh = (*integralHistogram)[iiOffset(xh+1,yh+1)];
        
        for (int i=0; i<weightDim; i++) {
          gradient[i] -= featureMap_xlyl[i] - featureMap_xlyh[i] - featureMap_xhyl[i] + featureMap_xhyh[i];
        }
        
        // compute expectation
        for (int i=0; i<weightDim; i++) {
          gradient[i] += expectation[i];
        }
      }
    
    }
  }
}


// SLIDING WINDOW FUNCTIONS

// sliding window using expectation
void PiecewiseGradient::slidingWindowExpectation(Dvector &expectation, int imageNumber)
{
  double p_plus, p_minus;
  int weightDim = expectation.size();

  Dvector expectation_plus(weightDim, 0.0);
  Dvector expectation_minus(weightDim, 0.0);
  Dvector expectation_west_plus(weightDim, 0.0);
  Dvector expectation_west_minus(weightDim, 0.0);
  Dvector expectation_north_plus(weightDim, 0.0);
  Dvector expectation_north_minus(weightDim, 0.0);
  Dvector expectation_east_plus(weightDim, 0.0);
  Dvector expectation_east_minus(weightDim, 0.0);
  Dvector expectation_south_plus(weightDim, 0.0);
  Dvector expectation_south_minus(weightDim, 0.0);
  Dvector expectation_northwest_plus(weightDim);
  Dvector expectation_northeast_plus(weightDim);
  Dvector expectation_northeast_minus(weightDim);
  Dvector expectation_southeast_plus(weightDim);
  Dvector expectation_southwest_plus(weightDim);
  Dvector expectation_southwest_minus(weightDim);
  
  // compute normalization constant
  Dvector logZ_F (4);
  crf->slidingWindowLogSumExp(logZ_F);

  IntegralImage *integralImage = crf->getIntegralImage();
  IntegralHistogram *integralHistogram = crf->getIntegralHistogram();
  
  // base expectation
  for (short y = 1; y < iiHeight - 1; y++) {
    for (short x = 1; x < iiWidth - 1; x++) {
      p_plus  = exp((*integralImage)[iiOffset(x,y)] - logZ_F[0]);
      p_minus = exp(-(*integralImage)[iiOffset(x,y)] - logZ_F[1]);
      Ivector &featureMap = (*integralHistogram)[iiOffset(x,y)];
      for (int i=0; i<weightDim; i++) {
        expectation_plus[i]  += p_plus*featureMap[i];
        expectation_minus[i] -= p_minus*featureMap[i];
      }
    }
  }
  
  // west
  {
    short x = 0;
    for (short y = 1; y < iiHeight - 1; y++) {
      p_plus  = exp((*integralImage)[iiOffset(x,y)] - logZ_F[0]);
      p_minus = exp(-(*integralImage)[iiOffset(x,y)] - logZ_F[1]);
      Ivector &featureMap = (*integralHistogram)[iiOffset(x,y)];
      for (int i=0; i<weightDim; i++) {
        expectation_west_plus[i]  += p_plus*featureMap[i];
        expectation_west_minus[i] -= p_minus*featureMap[i];
      }
    }
  }
  // north
  {
    short y = 0;
    for (short x = 1; x < iiWidth - 1; x++) {
      p_plus  = exp((*integralImage)[iiOffset(x,y)] - logZ_F[0]);
      p_minus = exp(-(*integralImage)[iiOffset(x,y)] - logZ_F[1]);
      Ivector &featureMap = (*integralHistogram)[iiOffset(x,y)];
      for (int i=0; i<weightDim; i++) {
        expectation_north_plus[i]  += p_plus*featureMap[i];
        expectation_north_minus[i] -= p_minus*featureMap[i];
      }
    }
  }
  // east
  {
    short x = iiWidth - 1;
    for (short y = 1; y < iiHeight - 1; y++) {
      p_plus  = exp((*integralImage)[iiOffset(x,y)] - logZ_F[0]);
      p_minus = exp(-(*integralImage)[iiOffset(x,y)] - logZ_F[1]);
      Ivector &featureMap = (*integralHistogram)[iiOffset(x,y)];
      for (int i=0; i<weightDim; i++) {
        expectation_east_plus[i]  += p_plus*featureMap[i];
        expectation_east_minus[i] -= p_minus*featureMap[i];
      }
    }
  }
  // south
  {
    short y = iiHeight - 1;
    for (short x = 1; x < iiWidth - 1; x++) {
      p_plus  = exp((*integralImage)[iiOffset(x,y)] - logZ_F[0]);
      p_minus = exp(-(*integralImage)[iiOffset(x,y)] - logZ_F[1]);
      Ivector &featureMap = (*integralHistogram)[iiOffset(x,y)];
      for (int i=0; i<weightDim; i++) {
        expectation_south_plus[i]  += p_plus*featureMap[i];
        expectation_south_minus[i] -= p_minus*featureMap[i];
      }
    }
  }
  // southwest
  {
    short x = 0;
    short y = iiHeight - 1;
    p_plus  = exp((*integralImage)[iiOffset(x,y)] - logZ_F[0]);
    p_minus = exp(-(*integralImage)[iiOffset(x,y)] - logZ_F[1]);
    Ivector &featureMap = (*integralHistogram)[iiOffset(x,y)];
    for (int i=0; i<weightDim; i++) {
      expectation_southwest_plus[i]  = p_plus*featureMap[i];
      expectation_southwest_minus[i] = -p_minus*featureMap[i];
    }
  }
  // northwest
  {
    short x = 0;
    short y = 0;
    p_plus  = exp((*integralImage)[iiOffset(x,y)] - logZ_F[0]);
    Ivector &featureMap = (*integralHistogram)[iiOffset(x,y)];
    for (int i=0; i<weightDim; i++) {
      expectation_northwest_plus[i] = p_plus*featureMap[i];
    }
  }
  // northeast
  {
    short x = iiWidth - 1;
    short y = 0;
    p_plus  = exp((*integralImage)[iiOffset(x,y)] - logZ_F[0]);
    p_minus = exp(-(*integralImage)[iiOffset(x,y)] - logZ_F[1]);
    Ivector &featureMap = (*integralHistogram)[iiOffset(x,y)];
    for (int i=0; i<weightDim; i++) {
      expectation_northeast_plus[i]  = p_plus*featureMap[i];
      expectation_northeast_minus[i] = -p_minus*featureMap[i];
    }
  }
  // southeast
  {
    short x = iiWidth - 1;
    short y = iiHeight - 1;
    p_plus  = exp((*integralImage)[iiOffset(x,y)] - logZ_F[0]);
    Ivector &featureMap = (*integralHistogram)[iiOffset(x,y)];
    for (int i=0; i<weightDim; i++) {
      expectation_southeast_plus[i] = p_plus*featureMap[i];
    }
  }
  

  // compute final result
  // expectation[i] = expectation_xlyl[i] + expectation_xlyh[i] + 
  //                  expectation_xhyl[i] + expectation_xhyh[i]
  double xlyh2xhyl = exp(logZ_F[1] - logZ_F[2]);
  double xlyl2xhyh = exp(logZ_F[0] - logZ_F[3]);
  for (int i=0; i<weightDim; i++) {
    expectation[i] = expectation_west_plus[i] + expectation_northwest_plus[i] + 
                       expectation_north_plus[i] + expectation_west_minus[i] + 
                       expectation_southwest_minus[i] + expectation_south_minus[i] +
                     (1 + xlyl2xhyh) * expectation_plus[i] + 
                     (1 + xlyh2xhyl) * expectation_minus[i] +
                     xlyl2xhyh * (expectation_east_plus[i] + 
                       expectation_southeast_plus[i] + expectation_south_plus[i]) +
                     xlyh2xhyl * (expectation_north_minus[i] + 
                       expectation_northeast_minus[i] + expectation_east_minus[i]);
  }
  
  /*
  // The above is possibly faster than this one
  // but this is easier to understand
  for (int i=0; i<weightDim; i++) {
    expectation[i] = expectation_plus[i] + expectation_west_plus[i] + 
                     expectation_northwest_plus[i] + expectation_north_plus[i]
                   + expectation_minus[i] + expectation_west_minus[i] + 
                     expectation_southwest_minus[i] + expectation_south_minus[i]
                   + exp(logZ_F[1] - logZ_F[2]) * (expectation_minus[i] + 
                     expectation_north_minus[i] + expectation_northeast_minus[i] + 
                     expectation_east_minus[i])
                   + exp(logZ_F[0] - logZ_F[3]) * (expectation_plus[i] + 
                     expectation_east_plus[i] + expectation_southeast_plus[i] + 
                     expectation_south_plus[i]);
  }
  */

}


