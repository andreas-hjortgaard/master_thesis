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

#include <cstdio>
#include <fstream>
#include <lbfgs.h>

#include "LBFGS.h"
#include "Types.h"

using namespace std;

// wrapper for using the libLBFGS C library
// constructor
LBFGS::LBFGS(ObjectiveFunction *obj, Gradient *grad) : GradientDescent(obj, grad), m_x(NULL) { 

  // initialize L-BFGS parameters
  // use Wolfe conditions
  lbfgs_parameter_init(&params);
  params.linesearch = LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
  params.epsilon    = 1e-3;     // stop criterion: ||g|| < epsilon * max(1, ||w||)
  params.past       = 10;       
  params.delta      = 1e-6;     // stop criterion: (f' - f) / f < delta 
                                // where f' is the objective value of past iterations ago

  function_evals = 0;
  gradient_evals = 0;
  
  iterations = 0;

  tempWeightsPath = "tempWeights.txt";

}

// deconstructor
LBFGS::~LBFGS() { 
  if (m_x != NULL) {
    lbfgs_free(m_x);
    m_x = NULL;
  }
}

// set path for storing temporary weights
void LBFGS::setTempWeightsPath(string path) {
  tempWeightsPath = path;
}

// get number of iterations 
int LBFGS::getIterations() {
  return iterations;
}

// learn weights using LBFGS optimization
// uses libLBFGS
Weights LBFGS::learnWeights(const Weights &w) {
  
  int ret, n;
  lbfgsfloatval_t fx;
  
  n = w.size();
  m_x = lbfgs_malloc(n);
  Weights w_new(n);

  // reset function and gradient evaluation counts
  function_evals = 0;
  gradient_evals = 0;

  // initialize variables to the weight values
  for (int i=0; i<n; i++) {
    m_x[i] = w[i];
  }
  
  // call L-BFGS procedure
  printf("Running LBFGS procedure...\n");
  fflush(stdout);
  ret = lbfgs(n, m_x, &fx, _evaluate, _progress, this, &params);
  
  printf("L-BFGS optimization terminated with status code = %d\n", ret);
  
  // convert result to Weights and return
  for (int i=0; i<n; i++) {
    w_new[i] = m_x[i];
  }
  
  return w_new;
}



// evaluate objective function and gradient
// converts weight vectors back and forth to arrays which is time consuming!
lbfgsfloatval_t LBFGS::evaluate(
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
        )
{
  
  Weights w(n);
  for (int i=0; i<n; i++) {
    w[i] = x[i];
  }
  
  Bboxes samples(10);
  
  // evaluate objective function
  double fx;
  fx = objective->evaluate(w);
  
  // compute gradient
  Dvector grad(w.size());

  gradient->evaluate(grad, w);
  
  for (int i=0; i<n; i++) {
    g[i] = grad[i];
  }
  
  function_evals++;
  gradient_evals++;

  return (lbfgsfloatval_t) fx;
}

// print progress after each iteration and store temporary weights
int LBFGS::progress(
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

  // TODO: maybe change to cout
  printf("Iteration %d:\n", k);
  printf("  obj = %f, w[0] = %f, w[1] = %f ...\n", fx, x[0], x[1]);
  printf("  wnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
  printf("  function evaluations: %d\n  gradient evaluations: %d\n", function_evals, gradient_evals);
  printf("\n"); 

  // update iterations class variable
  iterations = k;

  // store weights
  ofstream tempWeightFile(tempWeightsPath.c_str());
  if (!tempWeightFile) {
    cerr << "Could not open file " << tempWeightsPath << endl;
  }
  
  // store result in file
  for (int i=0; i<n; i++) {
    tempWeightFile << x[i] << "\n"; // endl is too slow for me!
  }
  
  tempWeightFile.close();
  
  fflush(stdout);

  return 0;
}


