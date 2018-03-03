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

#include "ConditionalRandomField.h"
#include "DataManager.h"

using namespace std;


// implementation of conditional random field

// constructor
ConditionalRandomField::ConditionalRandomField(DataManager *dataman) : 
  dataManager(dataman) { }



// setters and getters
void ConditionalRandomField::setDataManager(DataManager *dataman) {
  dataManager = dataman;
}

DataManager *ConditionalRandomField::getDataManager() {
  return dataManager;
}

void ConditionalRandomField::setWeights(Weights w) {
  weights = w;
  weightDim = (int) w.size();
}

Weights ConditionalRandomField::getWeights() {
  return weights;
}

int ConditionalRandomField::getWeightDim() {
  return weightDim;
}

void ConditionalRandomField::setStepSize(int stepSz) {
  stepSize = stepSz;
}


int ConditionalRandomField::getStepSize() {
  return stepSize;
}

int ConditionalRandomField::getNumImages() {
  return dataManager->getNumFiles();
}

IntegralImage *ConditionalRandomField::getIntegralImage() {
  return &integralImage;
}

IntegralHistogram *ConditionalRandomField::getIntegralHistogram() {
  return &integralHistogram;
}

int ConditionalRandomField::getIntegralImageWidth() {
  return iiWidth;
}

int ConditionalRandomField::getIntegralImageHeight() {
  return iiHeight;
}


// PROBABILITY FUNCTIONS
// probability of bounding box given image and class weights
double ConditionalRandomField::P(const Bbox &bbox, int imageNumber) {
  return P(bbox, imageNumber, weights);  
}

// probability of bounding box given image and weights
double ConditionalRandomField::P(const Bbox &bbox, int imageNumber, const Weights &w) {
  
  double dotproduct, logZ;  

  // compute integral image for the given image number
  computeIntegralImage(imageNumber, w);

  // compute score for the given bounding box
  dotproduct = computeBboxScore(bbox.ltrb[LEFT], bbox.ltrb[TOP], bbox.ltrb[RIGHT], bbox.ltrb[BOTTOM]);

  // compute normalization constant
  logZ = slidingWindowLogSumExp();
    
  return exp(dotproduct - logZ);
}


// conditional probability of one bbox coordinate given the rest
double ConditionalRandomField::condP(int var, const Bbox &bbox, int imageNumber, bool computeLogZ, double logZ) {
  return condP(var, bbox, imageNumber, weights, computeLogZ, logZ);
}

// computes the conditional probability of a single bbox variable given the rest
double ConditionalRandomField::condP(int var, const Bbox &box, int imageNumber, const Weights &w, bool computeLogZ, double logZ) {

  double dotproduct;

  if (computeLogZ) {
    // compute log of normalizing constant
    logZ = slidingWindowLogSumExpCond(var, box);
    // compute integral image for the given image number
    computeIntegralImage(imageNumber, w);
  }
  
  // compute score for the given bounding box
  dotproduct = computeBboxScore(box.ltrb[LEFT], box.ltrb[TOP], box.ltrb[RIGHT], box.ltrb[BOTTOM]);

  return exp(dotproduct - logZ);
}


// marginal probability of one corner (that is two connected sides) of the bbox
double ConditionalRandomField::cornerP(int xvar, int yvar, const Bbox &bbox, int imageNumber, const Weights &w, bool computeIIlogZ, double logZ, double maxScore) {
  
  double dotproduct;
  short xstart, xstop, ystart, ystop;
  int xSumOver, ySumOver;
  
  if (computeIIlogZ) {
    // compute integral image for the given image number
    computeIntegralImage(imageNumber, w);
    
    // compute normalization constant and save maxScore
    logZ = slidingWindowLogSumExp(&maxScore);
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
  int sumExpDotproduct = 0.0;
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
  return exp(log(sumExpDotproduct) - logZ);
  
}


/**
 * compute integral image
 *
 * Inspired by Christoph Lampert's Efficient Subwindow Search code
 * https://sites.google.com/a/christoph-lampert.com/work/software
 */ 
void ConditionalRandomField::computeIntegralImage(int imageNumber, const Weights &argweight) {

  // get (x,y,c),weight from the desired image
  Images &images = dataManager->getImages();
  Image img = images[imageNumber];
  int argnumpoints = img.numFeatures;
  short *argxpos = img.x;
  short *argypos = img.y;
  short *argclst = img.c;
  
  // ensure step size is not too large
  if (img.width < stepSize || img.height < stepSize) {
    throw STEP_SIZE_TOO_LARGE;
  }
 
  // transform (x,y,c),weight into integral image representation
  // (we add one for boundary conditions)
  this->iiWidth = img.width/stepSize + 1;
  this->iiHeight = img.height/stepSize + 1;
  
  // setup integral image
  integralImage.clear();
  integralImage.resize(iiWidth*iiHeight, 0.);
  
  // quantize all feature points and add to integral image
  // ignore extreme feature points in integral image
  short x,y,c;
  for (int k=0; k<argnumpoints; k++) {
    x = argxpos[k]/stepSize +1;
    y = argypos[k]/stepSize +1;
    c = argclst[k];
    if (x < iiWidth && y < iiHeight) {
      integralImage[iiOffset(x,y)] += argweight[c];
    }
  }
  
  // calculate integral image vertically
  for (short j=1; j < iiHeight; j++) {
    for (short i=1; i < iiWidth; i++) {
      integralImage[iiOffset(i,j)] += integralImage[iiOffset(i,j-1)];
    }
  }
  // calculate integral image horizontally
  for (short j=1; j < iiHeight; j++) {
    for (short i=1; i < iiWidth; i++) {
      integralImage[iiOffset(i,j)] += integralImage[iiOffset(i-1,j)];
    }
  }
}


// compute integral histogram
void ConditionalRandomField::computeIntegralHistogram(int imageNumber) {
  
  // get (x,y,c),weight from the desired image
  Images& images = dataManager->getImages();
  Image img = images[imageNumber];
  int argnumpoints = img.numFeatures;
  short *argxpos = img.x;
  short *argypos = img.y;
  short *argclst = img.c;
  
  // ensure step size is not too large
  if (img.width < stepSize || img.height < stepSize) {
    throw STEP_SIZE_TOO_LARGE;
  }
  
  // transform (x,y,c),weight into integral image representation
  // (we add one for boundary conditions)
  this->iiWidth = img.width/stepSize + 1;
  this->iiHeight = img.height/stepSize + 1;

  // set up integral histogram
  integralHistogram.clear();
  integralHistogram.resize(iiWidth*iiHeight, Ivector(weightDim, 0));
  
  short x,y,c;
  for (int k=0; k<argnumpoints; k++) {
    x = argxpos[k]/stepSize +1;
    y = argypos[k]/stepSize +1;
    c = argclst[k];
    if (x < iiWidth && y < iiHeight) {
      integralHistogram[iiOffset(x,y)][c] += 1;
    }
  }
  
  // calculate integral image vertically
  for (int j=1; j < iiHeight; j++) {
    for (int i=1; i < iiWidth; i++) {
      for (int c=0; c < weightDim; c++) {
        integralHistogram[iiOffset(i,j)][c] += integralHistogram[iiOffset(i,j-1)][c];
      }
    }
  }
  // calculate integral image horizontally
  for (int j=1; j < iiHeight; j++) {
    for (int i=1; i < iiWidth; i++) {
      for (int c=0; c < weightDim; c++) {
        integralHistogram[iiOffset(i,j)][c] += integralHistogram[iiOffset(i-1,j)][c];
      }
    }
  }
}


// sliding window using log of sum of exponentials
double ConditionalRandomField::slidingWindowLogSumExp(double* saveMaxScore)
{
  // outline for numerical stability:
  // - compute max of all computeBboxScores, call it alpha
  // - compute sum-of-exp as alpha + log(sum(exp(v-alpha)))
  double sum = 0.0;
  double score; 
  double maxScore = -999999.;

  int maxScoreStepSize = 1; // not too large! new stepsize will in fact be stepsize*maxScoreStepSize 

  // In the following, remember that width and height are actually +1
  //for all bounding heights
  for (short bbox_h = 0; bbox_h < iiHeight - 1; bbox_h += maxScoreStepSize) {
    //for all bounding widths
    for (short bbox_w = 0; bbox_w < iiWidth - 1; bbox_w += maxScoreStepSize) {
      //for all Top-Left y-coordinates
      for (short y = 0; y < iiHeight - bbox_h - 1; y += maxScoreStepSize) {
        //all Top-Left x-coordinates
        for (short x = 0; x < iiWidth - bbox_w - 1; x += maxScoreStepSize) {
          score = computeBboxScore(x, y, x+bbox_w, y+bbox_h);
          if (score > maxScore) {
            maxScore = score;
          }
        }
      }
    }
  }
  
  // return maxScore value if needed
  if (saveMaxScore != 0) {
    *saveMaxScore = maxScore;
  }

  // compute sum of exp
  //for all bounding heights
  for (short bbox_h = 0; bbox_h < iiHeight - 1; bbox_h++) {
    //for all bounding widths
    for (short bbox_w = 0; bbox_w < iiWidth - 1; bbox_w++) {
      //for all Top-Left y-coordinates
      for (short y = 0; y < iiHeight - bbox_h - 1; y++) {
        //all Top-Left x-coordinates
        for (short x = 0; x < iiWidth - bbox_w - 1; x++) {
          score = computeBboxScore(x, y, x+bbox_w, y+bbox_h);
          sum += exp(score - maxScore);
        }
      }
    }
  }
  
  // compute final result
  return maxScore + log(sum);

}

// sliding window using log of sum of exponentials (for conditional probabilities)
double ConditionalRandomField::slidingWindowLogSumExpCond(int var, const Bbox &bbox) {

  // outline for numerical stability:
  // - compute max of all computeBboxScores, call it alpha
  // - compute sum-of-exp as alpha + log(sum(exp(v-alpha)))
  double sum = 0.0;
  double score; 
  double maxScore = -999999.;
  short start, stop;

  // store original value
  int val = bbox.ltrb[var];

  // select the boundaries for the given variable
  switch (var) {
    case LEFT:
      // left goes from left edge to y_r
      start = 0;
      stop = bbox.ltrb[RIGHT];
      break;
    case TOP:
      // top goes from top edge to y_b
      start = 0;
      stop = bbox.ltrb[BOTTOM];
      break;
    case RIGHT:
      // right goes from y_l to right edge
      start = bbox.ltrb[LEFT];
      stop  = iiWidth-2;  
      break;
    case BOTTOM:
      // bottom goes from y_t to bottom edge
      start = bbox.ltrb[TOP];
      stop  = iiHeight-2;
      break; 
  }


  // compute max score
  for (short i = start; i <= stop; i++) {
    bbox.ltrb[var] = i;
    score = computeBboxScore(bbox.ltrb[LEFT], bbox.ltrb[TOP], bbox.ltrb[RIGHT], bbox.ltrb[BOTTOM]);
    if (score > maxScore) {
      maxScore = score;
    }
  }

  // compute sum of exp
  for (short i = start; i <= stop; i++) {
    bbox.ltrb[var] = i;
    score = computeBboxScore(bbox.ltrb[LEFT], bbox.ltrb[TOP], bbox.ltrb[RIGHT], bbox.ltrb[BOTTOM]);
    sum += exp(score - maxScore);
  }

  // reinsert original value  
  bbox.ltrb[var] = val;

  // compute final result
  return maxScore + log(sum);

}


// sliding window -- func is e.g. sum or max
void ConditionalRandomField::slidingWindow(
  Bbox& result,
  void (*slidingFunc)(Bbox &, double, short, short, short, short),
  bool rescale)
{
  double candidateScore;
  // In the following, remember that width and height are actually +1
  //for all bounding heights
  for (short bbox_h = 0; bbox_h < iiHeight - 1; bbox_h++) {
    //for all bounding widths
    for (short bbox_w = 0; bbox_w < iiWidth - 1; bbox_w++) {
      //for all Top-Left y-coordinates
      for (short y = 0; y < iiHeight - bbox_h - 1; y++) {
        //all Top-Left x-coordinates
        for (short x = 0; x < iiWidth - bbox_w - 1; x++) {
          candidateScore = computeBboxScore(x, y, x+bbox_w, y+bbox_h);
          slidingFunc(result, candidateScore, x, y, x+bbox_w, y+bbox_h);
        }
      }
    }
  }
  if (rescale) {
    // rescale resulting bounding box
    result.ltrb[LEFT] *= stepSize;
    result.ltrb[TOP] *= stepSize;
    result.ltrb[RIGHT] = (result.ltrb[RIGHT]+1)*stepSize-1;
    result.ltrb[BOTTOM] = (result.ltrb[BOTTOM]+1)*stepSize-1;
  }
}

// functions for sliding window
void ConditionalRandomField::slidingMax(Bbox &maxBbox, double score, short xl, short yl, short xh, short yh) {
  if (score > maxBbox.score) {
    maxBbox.ltrb[LEFT] = xl;
    maxBbox.ltrb[TOP] = yl;
    maxBbox.ltrb[RIGHT] = xh;
    maxBbox.ltrb[BOTTOM] = yh;
    maxBbox.score = score;
  }
}

void ConditionalRandomField::slidingSum(Bbox &sumBbox, double score, short xl, short yl, short xh, short yh) {
  sumBbox.score += score;
}

