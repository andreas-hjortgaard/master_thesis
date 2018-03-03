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

#ifndef _GRADIENT_H_
#define _GRADIENT_H_ 

#include "DataManager.h"
#include "ConditionalRandomField.h"

// objective function base class
// defines structure and parameters for the objective function
class Gradient {

  protected:

    DataManager *dataManager;
    ConditionalRandomField *crf;

    // search index
    SearchIx searchIx;
    
    // regularizer
    double lambda;

    // for current integral image and integral histogram
    int iiWidth, iiHeight;


  public:

    // constructor
    Gradient(DataManager *dm=NULL, ConditionalRandomField *crfield=NULL, SearchIx si=SearchIx());

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
    
    IntegralHistogram *getIntegralHistogram();
    int getIntegralImageWidth();
    int getIntegralImageHeight();
       
    // useful functions which will be called through the CRF
    void computeIntegralImage(int imageNumber, Weights &w);
    void computeIntegralHistogram(int imageNumber);   
    double computeBboxScore(short xl, short yl, short xh, short yh);
    void computeFeatureMap(Ivector &featureMap, short xl, short yl, short xh, short yh);
    double slidingWindowLogSumExp();   
    int iiOffset(int x, int y);

    // evaluate (specific to the actual gradient)
    virtual void evaluate(Dvector &gradient, Weights &w, bool normalized = true) = 0;

};


inline double Gradient::computeBboxScore(short xl, short yl, short xh, short yh) {
  return crf->computeBboxScore(xl, yl, xh, yh);
}

inline void Gradient::computeFeatureMap(Ivector &featureMap, short xl, short yl, short xh, short yh) {
  crf->computeFeatureMap(featureMap, xl, yl, xh, yh);
}

// convert (x,y) into 1d index
inline int Gradient::iiOffset(int x, int y) {
  return crf->iiOffset(x,y);
}

#endif // _GRADIENT_H_
