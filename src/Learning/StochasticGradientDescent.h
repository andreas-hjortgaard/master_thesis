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

#ifndef _STOCHASTIC_GRADIENT_DESCENT_H_
#define _STOCHASTIC_GRADIENT_DESCENT_H_

#include "GradientDescent.h"
#include "ObjectiveFunctions/StochasticGradient.h"


// implementation of stochastic gradient descent
// inspired by Leon Bottou's crfsgd
// http://leon.bottou.org/projects/sgd
class StochasticGradientDescent : public GradientDescent {
  
  protected:

    StochasticGradient *gradient;

    // parameters for learning rate
    // eta(t) = 1/(alpha*(t+t0))    decreasing eta
    // eta(t) = 1/(alpha*t0)        constant eta
    bool constLearningRate;    
    double alpha;
    double t0;
      
    // maximum number of epochs
    int maxEpochs;
    
    // current epoch and iteration
    int epoch;
    int t; 
    
    // path for temporary weights
    std::string tempWeightsPath;

    // try given learning rate on sample subset
    // for use in initializeLearningRate
    double tryEta(Weights &w, double eta, SearchIx &sample, bool normalized);

    // print progress
    void progress(Weights &w, Weights &wAvg, int epoch, int t, double eta, bool lastEpoch) ;

  public:

    // constructors
    StochasticGradientDescent(ObjectiveFunction *obj, StochasticGradient *grad);
    StochasticGradientDescent(ObjectiveFunction *obj, StochasticGradient *grad, double alpha_, double t0_, bool constLearningRate = false);

    // getters and setters for learning rate parameters
    double getAlpha();
    void setAlpha(double alpha_);
    
    double getT0();
    void setT0(double t0_);
    
    void setMaxEpochs(int maxEpochs_);
    
    // path for storing temporary weights between iterations
    void setTempWeightsPath(std::string path);

    // use training data (or a subset) to initialize t0 and alpha
    void initializeLearningRate(Weights &w, double initialEta, int sampleSize, bool normalized);
    
    // training procedure
    virtual Weights learnWeights(const Weights &w);

};

#endif // _STOCHASTIC_GRADIENT_DESCENT_H_
