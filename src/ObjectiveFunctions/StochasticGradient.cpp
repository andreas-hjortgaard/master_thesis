// implementation of stochatic gradient
#include "StochasticGradient.h"

using namespace std;

// constructor
StochasticGradient::StochasticGradient(DataManager *dm, ConditionalRandomField *crfield, SearchIx si)
  : LogLikelihoodGradient::LogLikelihoodGradient(dm, crfield, si)
{
}

// stochastic gradient
// use log-likelihood gradient by simply setting the search index
void StochasticGradient::evaluate(Dvector &gradient, Weights &w, int imageNumber, bool normalized) {
  LogLikelihoodGradient::setSearchIx(imageNumber);
  LogLikelihoodGradient::evaluate(gradient, w, normalized);
}

