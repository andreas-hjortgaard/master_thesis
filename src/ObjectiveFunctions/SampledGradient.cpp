// implementation of stochatic gradient
#include <cstdio>
#include <cmath>
#include <iostream>

#include "SampledGradient.h"

using namespace std;

// constructor
SampledGradient::SampledGradient(DataManager *dm, ConditionalRandomField *crfield, GibbsSampler *gibbs, SearchIx si, int numSamples_, int skip_)
  : StochasticGradient::StochasticGradient(dm, crfield, si), sampler(gibbs), numSamples(numSamples_), skip(skip_)
{
}

// get and set sampler
GibbsSampler *SampledGradient::getSampler() {
  return sampler;
}
void SampledGradient::setSampler(GibbsSampler *gibbs) {
  sampler = gibbs;
}

int SampledGradient::getNumSamples() {
  return numSamples;
}

void SampledGradient::setNumSamples(int n) {
  numSamples = n;
}

int SampledGradient::getSkip() {
  return skip;
}

void SampledGradient::setSkip(int s) {
  skip = s;
}

// gradient evaluated for a single image
void SampledGradient::evaluate(Dvector &gradient, Weights &w, int imageNumber, bool normalized) {
  setSearchIx(imageNumber);
  evaluate(gradient, w, normalized);
}


// gradient (similar to LogLikelihoodGradient, except it uses sample mean instead of 
// real expectation
void SampledGradient::evaluate(Dvector &gradient, Weights &w, bool normalized) {

  int imageNumber;

  int weightDim = w.size();
  int stepSize = getStepSize(); // retrieve step size from CRF
  
  // check gradient size
  if (gradient.size() != w.size()) {
    throw GRADIENT_SIZE_ERROR;
  }
  
  Bboxes &bboxes = dataManager->getBboxes();

  // create dummy scaled bbox
  Bbox scaledBbox;
  scaledBbox.ltrb = new short[4];

  // create temporary Dvector
  Ivector featureMap      = Ivector(weightDim, 0);
  Ivector tempFeatureMap  = Ivector(weightDim, 0);
  Dvector sampleMean      = Dvector(weightDim, 0.0);
  
  // compute gradient regularizer
  // (no need to reset gradient before computation, just overwrite!)
  for (int i=0; i<weightDim; i++) {
    gradient[i] = 2*lambda*w[i];
  }

  // compute feature map and sample mean
  for (size_t j=0; j<searchIx.size(); j++) {

    // extract image index
    imageNumber = searchIx[j];

    // only compute integral image and integral histogram when necessary
    if (bboxes[imageNumber].numObject > 0) {

      // compute integral image
      computeIntegralImage(imageNumber, w);
      
      // compute integral histogram
      computeIntegralHistogram(imageNumber);
    }

    // only work on images that actually contain the object
    for (int numObj = 0; numObj < bboxes[imageNumber].numObject; numObj++) {
    
      // compute feature map
      // fit to quantized integralImage space
   
      scaledBbox.ltrb[LEFT]    = min(bboxes[imageNumber].ltrb[4*numObj+0]/stepSize, iiWidth-2);
      scaledBbox.ltrb[TOP]     = min(bboxes[imageNumber].ltrb[4*numObj+1]/stepSize, iiHeight-2);
      scaledBbox.ltrb[RIGHT]   = min(bboxes[imageNumber].ltrb[4*numObj+2]/stepSize, iiWidth-2);
      scaledBbox.ltrb[BOTTOM]  = min(bboxes[imageNumber].ltrb[4*numObj+3]/stepSize, iiHeight-2);

      computeFeatureMap(featureMap, scaledBbox.ltrb[LEFT], 
                                    scaledBbox.ltrb[TOP],
                                    scaledBbox.ltrb[RIGHT],
                                    scaledBbox.ltrb[BOTTOM]);
      
      // compute sample mean
      if (normalized) {
        computeSampleMean(sampleMean, w, tempFeatureMap, imageNumber, scaledBbox);
      }
      
      
      // update gradient
      for (int i=0; i<weightDim; i++) {
        gradient[i] -= featureMap[i] - sampleMean[i];
      }
    }
  }
  
  // delete dummy bbox
  delete[] scaledBbox.ltrb;  
}



// approximate expectation by sample mean
void SampledGradient::computeSampleMean(Dvector &sampleMean, Weights &w, Ivector &featureMap, int imageNumber, Bbox &bbox)
{
  int weightDim = sampleMean.size();

  // clear and reset sample mean
  sampleMean.clear();
  sampleMean.resize(weightDim, 0.0);

  Bbox *sample;

  // initialize Gibbs chain to ground truth bbox (scaled)
  sampler->initializeCurrent(imageNumber, bbox);

  // take number of samples
  for (int i=0; i<numSamples; i++) {
    sampler->sample(skip, w, imageNumber, false);   // don't recompute integral image
    sample = sampler->getCurrentSample(imageNumber);
    computeFeatureMap(featureMap, sample->ltrb[LEFT], sample->ltrb[TOP], sample->ltrb[RIGHT], sample->ltrb[BOTTOM]);

    // sum up feature maps   
    for (int j=0; j<weightDim; j++) {
      sampleMean[j] += featureMap[j];
    }
  }

  // divide by number of samples (compute sample mean)
  for (int j=0; j<weightDim; j++) {
    sampleMean[j] /= numSamples;
  }
}
