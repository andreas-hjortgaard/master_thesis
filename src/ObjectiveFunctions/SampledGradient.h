#ifndef _SAMPLED_GRADIENT_H_
#define _SAMPLED_GRADIENT_H_

#include "StochasticGradient.h"
#include "Inference/GibbsSampler.h"

// sampled gradient derived from the stochastic class
class SampledGradient : public StochasticGradient {

  private:
    
    // sampler for sampling the sample mean
    GibbsSampler *sampler;

    // number of samples, number of skips
    // number of skips corresponds to k in CD-k
    int numSamples;
    int skip;

    // computing the sample mean instead of expectation
    void computeSampleMean(Dvector &sampleMean, Weights &w, Ivector &featureMap, int imageNumber, Bbox &bbox);

  public:
  
    // constructor
    SampledGradient(DataManager *dm=NULL, ConditionalRandomField *crfield=NULL, GibbsSampler *sampler=NULL, SearchIx si=SearchIx(), int numSamples=10, int skip=10);

    GibbsSampler *getSampler();
    void setSampler(GibbsSampler *gibbs);

    int getNumSamples();
    void setNumSamples(int n);

    int getSkip();
    void setSkip(int s);

    // evaluate for a single training example
    virtual void evaluate(Dvector &gradient, Weights &w, int imageNumber, bool normalized = true);

    // evaluate sampled gradient
    virtual void evaluate(Dvector &gradient, Weights &w, bool normalized = true);


};

#endif // _SAMPLED_GRADIENT_H_

