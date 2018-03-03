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

#include <cmath>
#include "PiecewiseConditionalRandomField.h"

// implementation of piecewise conditional random field
using namespace std;

// constructor
PiecewiseConditionalRandomField::PiecewiseConditionalRandomField(DataManager *dataman) : 
  ConditionalRandomField(dataman) { }

// sliding window using log of sum of exponentials
void PiecewiseConditionalRandomField::slidingWindowLogSumExp(Dvector &logZ_F)
{
  // outline for numerical stability:
  // - compute max of all computeBboxScores, call it alpha
  // - compute sum-of-exp as alpha + log(sum(exp(v-alpha)))
  double score;
  double maxScore_plus  = -999999.;
  double maxScore_minus = -999999.;
  short maxScoreStepSize = 1; // not too large! new stepsize will in fact be stepsize*maxScoreStepSize 

  double sum_plus            = 0.0;
  double sum_minus           = 0.0;
  double sum_west_plus       = 0.0;
  double sum_west_minus      = 0.0;
  double sum_north_plus      = 0.0;
  double sum_north_minus     = 0.0;
  double sum_east_plus       = 0.0;
  double sum_east_minus      = 0.0;
  double sum_south_plus      = 0.0;
  double sum_south_minus     = 0.0;
  double sum_northwest_plus;
  double sum_northeast_minus;
  double sum_southeast_plus;
  double sum_southwest_minus;  

  // find max score
  for (short y = 0; y < iiHeight; y += maxScoreStepSize) {
    for (short x = 0; x < iiWidth; x += maxScoreStepSize) {
      score = integralImage[iiOffset(x,y)];
      if (score > maxScore_plus) {
        maxScore_plus = score;
      }
      if (-score > maxScore_minus) {
        maxScore_minus = -score;
      }
    }
  }
  
  // compute sum of exp
  // base sum
  for (short y = 1; y < iiHeight - 1; y++) {
    for (short x = 1; x < iiWidth - 1; x++) {
      score = integralImage[iiOffset(x,y)];
      sum_plus  += exp(score - maxScore_plus);
      sum_minus += exp(-score - maxScore_minus);
    }
  }
  // west
  {
    short x = 0;
    for (short y = 1; y < iiHeight - 1; y++) {
      score = integralImage[iiOffset(x,y)];
      sum_west_plus  += exp(score - maxScore_plus);
      sum_west_minus += exp(-score - maxScore_minus);
    }
  }
  // north
  {
    short y = 0;
    for (short x = 1; x < iiWidth - 1; x++) {
      score = integralImage[iiOffset(x,y)];
      sum_north_plus  += exp(score - maxScore_plus);
      sum_north_minus += exp(-score - maxScore_minus);
    }
  }
  // east
  {
    short x = iiWidth - 1;
    for (short y = 1; y < iiHeight - 1; y++) {
      score = integralImage[iiOffset(x,y)];
      sum_east_plus  += exp(score - maxScore_plus);
      sum_east_minus += exp(-score - maxScore_minus);
    }
  }
  // south
  {
    short y = iiHeight - 1;
    for (short x = 1; x < iiWidth - 1; x++) {
      score = integralImage[iiOffset(x,y)];
      sum_south_plus  += exp(score - maxScore_plus);
      sum_south_minus += exp(-score - maxScore_minus);
    }
  }
  // southwest
  {
    short x = 0;
    short y = iiHeight - 1;
    score = integralImage[iiOffset(x,y)];
    sum_southwest_minus = exp(-score - maxScore_minus);
  }
  // northwest
  {
    short x = 0;
    short y = 0;
    score = integralImage[iiOffset(x,y)];
    sum_northwest_plus = exp(score - maxScore_plus);
  }
  // northeast
  {
    short x = iiWidth - 1;
    short y = 0;
    score = integralImage[iiOffset(x,y)];
    sum_northeast_minus = exp(-score - maxScore_minus);
  }
  // southeast
  {
    short x = iiWidth - 1;
    short y = iiHeight - 1;
    score = integralImage[iiOffset(x,y)];
    sum_southeast_plus = exp(score - maxScore_plus);
  }
  
  // compute final result
  // xl,yl
  logZ_F[0] = maxScore_plus  + log(sum_plus + sum_west_plus + sum_northwest_plus + sum_north_plus);
  // xl,yh
  logZ_F[1] = maxScore_minus + log(sum_minus + sum_west_minus + sum_southwest_minus + sum_south_minus);
  // xh,yl
  logZ_F[2] = maxScore_minus + log(sum_minus + sum_north_minus + sum_northeast_minus + sum_east_minus);
  // xh,yh
  logZ_F[3] = maxScore_plus  + log(sum_plus + sum_east_plus + sum_southeast_plus + sum_south_plus);
  
  return;

}


// marginal probability of one corner (that is two connected sides) of the bbox
double PiecewiseConditionalRandomField::cornerP(int xvar, int yvar, const Bbox &bbox, int imageNumber, const Weights &w, bool computeIIlogZ, Dvector logZ_F, double maxScore) {
  
  double dotproduct;
  short xstart, xstop, ystart, ystop;
  int xSumOver, ySumOver;
  
  if (computeIIlogZ) {
    // compute integral image for the given image number
    computeIntegralImage(imageNumber, w);
    
    // compute normalization constant and save maxScore
    slidingWindowLogSumExp(logZ_F);
  }
  
  // select the boundaries for the variables to be summed over
  switch (xvar) {
    case LEFT:
      xSumOver = RIGHT;
      // right goes from y_l to right edge
      xstart = bbox.ltrb[LEFT];
      xstop  = iiWidth-2;  
      break;
    case RIGHT:
      xSumOver = LEFT;
      // left goes from left edge to y_r
      xstart = 0;
      xstop = bbox.ltrb[RIGHT];
      break;
  }
  
  switch (yvar) {
    case TOP:
      ySumOver = BOTTOM;
      // bottom goes from y_t to bottom edge
      ystart = bbox.ltrb[TOP];
      ystop  = iiHeight-2;
      break;
      
    case BOTTOM:
      ySumOver = TOP;
      // top goes from top edge to y_b
      ystart = 0;
      ystop = bbox.ltrb[BOTTOM];
      break;
  }
  
  // store original values
  int xval = bbox.ltrb[xSumOver];
  int yval = bbox.ltrb[ySumOver];

  // compute sum of exp of dotproducts
  // For numerical stability: NO! does not work! //maxScore + log(sum(exp(v-maxScore)))
  double sumExpDotproduct = 0.0;
  for (short j = ystart; j <= ystop; j++) {
    bbox.ltrb[ySumOver] = j;
    for (short i = xstart; i <= xstop; i++) {
      bbox.ltrb[xSumOver] = i;
      dotproduct = computeBboxScore(bbox.ltrb[LEFT], bbox.ltrb[TOP], bbox.ltrb[RIGHT], bbox.ltrb[BOTTOM]);
      sumExpDotproduct += exp(dotproduct);
    }
  }

  // reinsert original values
  bbox.ltrb[xSumOver] = xval;
  bbox.ltrb[ySumOver] = yval;

  // compute final result
  return exp(log(sumExpDotproduct) - logZ_F[0] - logZ_F[1] - logZ_F[2] - logZ_F[3]);
  
}
