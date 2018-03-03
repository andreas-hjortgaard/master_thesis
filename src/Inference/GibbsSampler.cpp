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

// implementation of Gibbs sampler
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "GibbsSampler.h"

using namespace std;

// constructors 
GibbsSampler::GibbsSampler(ConditionalRandomField *crf_) :  
  crf(crf_)
{
  srand(time(NULL)); rand();
  initialize();
} 

GibbsSampler::GibbsSampler(ConditionalRandomField *crf_, Bboxes initial) :  
  crf(crf_),
  current(initial)
{ 
  srand(time(NULL)); rand();
  initialize();
  
  for (size_t i=0; initial.size(); i++) {
    initializeCurrent(i, initial[i]);
  }
} 


GibbsSampler::~GibbsSampler() {

  // if current has been initialized, free memory
  if (!current.empty()) {
    for (size_t i=0; i<current.size(); i++) {
      delete[] current[i].ltrb;
    }
  }
}

// initialize Gibbs chain with random bboxes
void GibbsSampler::initialize() {

  int numImages = crf->getNumImages();
  int stepSize = crf->getStepSize();  

  current.clear();
  current.resize(numImages);
  step.clear();
  step.resize(numImages, 0);

  // initialize current bbox to random
  for (int i=0; i<numImages; i++) { 
 
    // get real image sizes
    int imageWidth  = crf->getDataManager()->getImages()[i].width;
    int imageHeight = crf->getDataManager()->getImages()[i].height;
    
    // compute integral image sizes
    int iiWidth   = imageWidth/stepSize+1;
    int iiHeight  = imageHeight/stepSize+1;
    short left, top, right, bottom;
    
    // select random bbox 
    // left   = [0, iiWidth-2]
    // top    = [0, iiHeight-2]
    // right  = [left, iiWidth-2]
    // bottom = [top, iiHeight-2]
    left    = rand() % (iiWidth-1);
    top     = rand() % (iiHeight-1);
    right   = left + (rand()%(iiWidth-1-left));
    bottom  = top + (rand()%(iiHeight-1-top));
  
    current[i].ltrb = new short[4];
    current[i].ltrb[LEFT]   = left;
    current[i].ltrb[TOP]    = top; 
    current[i].ltrb[RIGHT]  = right;
    current[i].ltrb[BOTTOM] = bottom;
  
  }
}

// initialize current sample for given image with a given bbox
void GibbsSampler::initializeCurrent(int imageNumber, Bbox &bbox) {

  current[imageNumber].ltrb[LEFT]    = bbox.ltrb[LEFT];
  current[imageNumber].ltrb[TOP]     = bbox.ltrb[TOP];
  current[imageNumber].ltrb[RIGHT]   = bbox.ltrb[RIGHT];
  current[imageNumber].ltrb[BOTTOM]  = bbox.ltrb[BOTTOM];

  step[imageNumber] = 0;
  
}


// return current state of the Gibbs chain
Bboxes *GibbsSampler::getCurrentSample() {
  return &current;
}

// return current state of the Gibbs chain for a given image
Bbox *GibbsSampler::getCurrentSample(int imageNumber) {
  return &current[imageNumber];
}

// return cumulative histogram
Dvector GibbsSampler::getCumulativeHistogram() {
  return cumulativeHistogram;  
}



// compute cumulative histogram (var is 0,1,2,3 corresponing to left, top, right, bottom)
void GibbsSampler::computeCumulativeHistogram(int var, Weights &w, int imageNumber) {

  // Outline: 
  // - fix three variables, say y_t, y_r, y_b
  // - compute probability of each value of the free variable, y_l
  // - sum up probabilities for in cumulative histogram 

  double prob, logZ;
  int hSize;

  int iiWidth = crf->getIntegralImageWidth();
  int iiHeight = crf->getIntegralImageHeight();

  // make space for histogram
  switch (var) {
    case LEFT:
      histogramOffset = 0;
      hSize = current[imageNumber].ltrb[RIGHT]+1;
      break;
    case TOP:
      histogramOffset = 0;
      hSize = current[imageNumber].ltrb[BOTTOM]+1;
      break;
    case RIGHT:
      histogramOffset = current[imageNumber].ltrb[LEFT];
      hSize = iiWidth - histogramOffset - 1;            
      break;
    case BOTTOM:
      histogramOffset = current[imageNumber].ltrb[TOP];
      hSize = iiHeight - histogramOffset - 1;
      break;
  }

  // compute cumulative histogram
  Bbox &bbox = current[imageNumber];

  // store old value
  int val = bbox.ltrb[var];
  
  // clear and resize histogram to hold new values
  cumulativeHistogram.clear();
  cumulativeHistogram.resize(hSize, 0.0);

  // compute logZ
  logZ = crf->slidingWindowLogSumExpCond(var, bbox);

  // initialize first value in histogram
  bbox.ltrb[var] = histogramOffset;
  cumulativeHistogram[0] = crf->condP(var, bbox, imageNumber, w, false, logZ);

  // compute the rest
  for (int i=1; i<hSize; i++) {
    bbox.ltrb[var] = i+histogramOffset;
    prob = crf->condP(var, bbox, imageNumber, w, false, logZ);
      
    cumulativeHistogram[i] = cumulativeHistogram[i-1] + prob;
  }
  
  // reinsert original value  
  bbox.ltrb[var] = val;

} 

// sample one variable from conditional distribution 
// using the inverse transform (Smirnov) method
void GibbsSampler::sampleOne(int var, Weights &w, int imageNumber) {

  double randnum; 
  int i, sample ; 

  // compute cumulative histogram
  computeCumulativeHistogram(var, w, imageNumber);

  // draw a random number between 0 and 1
  randnum = ((double) rand() / RAND_MAX);

  // draw sample from the cumulative distribution
  // by searching through histogram
  i = 0;
  while (cumulativeHistogram[i] < randnum && i<(int)cumulativeHistogram.size()) {
    i++;
  }

  // choose desired sample
  sample = i+histogramOffset;

  // update current sample
  current[imageNumber].ltrb[var] = sample;

}   

// take k steps of the Gibbs chain to obtain one sample
void GibbsSampler::sample(int k, Weights &w, int imageNumber, bool computeII) {
  
  // compute integral image
  if (computeII) {
    crf->computeIntegralImage(imageNumber, w);
  }
  
  // run k Gibbs steps
  for (int i=0; i<k; i++) {
    
    // sample from one variable at a time
    sampleOne(LEFT, w, imageNumber);
    sampleOne(TOP, w, imageNumber);
    sampleOne(RIGHT, w, imageNumber);
    sampleOne(BOTTOM, w, imageNumber);
    step[imageNumber]++;

  }
}

