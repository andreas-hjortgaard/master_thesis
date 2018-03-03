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

#ifndef _OBJECTIVE_FUNCTION_H_
#define _OBJECTIVE_FUNCTION_H_ 

#include "DataManager.h"
#include "ConditionalRandomField.h"


// objective function base class
// defines structure and parameters for the objective function
class ObjectiveFunction {

  protected:

    // has classes for data and distribution
    DataManager *dataManager;
    ConditionalRandomField *crf;
    
    // search index
    SearchIx searchIx;

    // regularization constant
    double lambda;
    
    // for current integral image
    int iiWidth, iiHeight;

  public:

    // constructor
    ObjectiveFunction(DataManager *dm=NULL, ConditionalRandomField *crf=NULL, SearchIx si=SearchIx());

    // getters/setters
    DataManager *getDataManager();
    void setDataManager(DataManager *dm);

    ConditionalRandomField *getCRF();
    void setCRF(ConditionalRandomField *crf);
  
    SearchIx getSearchIx();
    void setSearchIx(SearchIx si);
    void setSearchIx(int i);        // use single example
    
    int getStepSize();

    // get and set regularization constant
    double getLambda();
    void setLambda(double lambda);

    int getNumImages();
    SearchIx getNonEmpty();
    
    // useful functions which will be called through the CRF
    void computeIntegralImage(int imageNumber, Weights &w);

    double computeBboxScore(short xl, short yl, short xh, short yh);

    double slidingWindowLogSumExp();

    
    IntegralImage *getIntegralImage();
    short getIntegralImageWidth();
    short getIntegralImageHeight();

    // evaluate (specific to the actual objective function)
    virtual double evaluate(Weights &w, bool normalized = true) = 0;

};


inline double ObjectiveFunction::computeBboxScore(short xl, short yl, short xh, short yh) {
  return crf->computeBboxScore(xl, yl, xh, yh);
}


#endif // _OBJECTIVE_FUNCTION_H_
