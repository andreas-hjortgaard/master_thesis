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

#ifndef _LBFGS_H_
#define _LBFGS_H_

#include <string>
#include <lbfgs.h>
#include "GradientDescent.h"

// LBFGS is a wrapper for the libLBFGS library
// uses x   for weights when used by liblbfgs
// uses fx  for function value
// uses g   for gradient
// uses n   for weight dimensions
// uses k   for iteration
class LBFGS : public GradientDescent {

  private:

    int function_evals;
    int gradient_evals;

    int iterations;

    // string for temporary weights
    std::string tempWeightsPath;

  protected:
    
    lbfgsfloatval_t *m_x;
    lbfgs_parameter_t params;
    
    // functions for evaluating objective function
   static lbfgsfloatval_t _evaluate(
        void *instance,
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
        )
    {
      return reinterpret_cast<LBFGS*>(instance)->evaluate(x, g, n, step);
    }
    
    lbfgsfloatval_t evaluate(
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
        );
        
    // functions for progress reporting
    static int _progress(
        void *instance,
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
        )
    {
      return reinterpret_cast<LBFGS*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
    }
        
    int progress(
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
        );

  public:
    
    // constructor
    LBFGS(ObjectiveFunction *obj, Gradient *grad);
    ~LBFGS();

    // path for storing temporary weights between iterations
    void setTempWeightsPath(std::string path);
    
    // get number of iterations 
    int getIterations();
  
    // redefine learnWeights function
    virtual Weights learnWeights(const Weights &w);
    
};


#endif // _LBFGS_H_
