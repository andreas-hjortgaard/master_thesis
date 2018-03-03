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
 
#ifndef _CONDITIONAL_RANDOM_FIELD_H_
#define _CONDITIONAL_RANDOM_FIELD_H_

#include "DataManager.h"

// Conditional Random Field class
// for computing probabilities of bounding boxes
// as well as conditional probabilties of the individual box coordinates
class ConditionalRandomField {

  protected:
    
    // data manager for holding image datasets
    DataManager *dataManager;

    // weight vector and dimensionality
    Weights weights;
    int weightDim;

    // integral image for computing bbox score
    // integral histogram for computing feature map
    // stepSize denotes the quantization
    IntegralImage integralImage;
    IntegralHistogram integralHistogram;
    int iiWidth, iiHeight;
    int stepSize;


  public:
    
    // constructor
    ConditionalRandomField(DataManager *dataman=NULL);

    // setters and getters
    void setDataManager(DataManager *dataman);
    DataManager *getDataManager();

    void setWeights(Weights weights);
    Weights getWeights();

    int getWeightDim();

    void setStepSize(int stepSz);
    int getStepSize();

    int getNumImages();

    IntegralImage *getIntegralImage();
    IntegralHistogram *getIntegralHistogram();
    int getIntegralImageWidth();
    int getIntegralImageHeight();

    // PROBABILITY FUNCTIONS
    // probability of bounding box given image and class weights
    double P(const Bbox &bbox, int imageNumber);

    // probabilty of bounding box given image and weights
    double P(const Bbox &bbox, int imageNumber, const Weights &w);
   
    // conditional probability of one bbox coordinate given the rest
    double condP(int var, const Bbox &bbox, int imageNumber, bool computeLogZ = true, double logZ = 0.0);
    double condP(int var, const Bbox &bbox, int imageNumber, const Weights &w, bool computeLogZ = true, double logZ = 0.0);

    // marginal probability of one corner (that is two connected sides) of the bbox
    double cornerP(int xvar, int yvar, const Bbox &bbox, int imageNumber, const Weights &w, bool computeIIlogZ = true, double logZ = 0.0, double maxScore = 0.0);


    // HELPER FUNCTIONS FOR COMPUTING PROBABILITIES

    /**
     * Calculate score of a box from integral image
     * A bounding box includes the (xl,yl,xh,yh) coordinates
     * computeBboxScore works directly in the quantized space
     *
     * Inspired by Christoph Lampert's Efficient Subwindow Search code
     * https://sites.google.com/a/christoph-lampert.com/work/software
     */ 
    double computeBboxScore(short xl, short yl, short xh, short yh);

    // compute feature map given any y
    void computeFeatureMap(Ivector &featureMap, short xl, short yl, short xh, short yh);

    // convert (x,y) into 1d index
    int iiOffset(int x, int y);

 
    /**
     * compute integral image
     *
     * Inspired by Christoph Lampert's Efficient Subwindow Search code
     * https://sites.google.com/a/christoph-lampert.com/work/software
     */ 
    void computeIntegralImage(int imageNumber, const Weights &argweight);

    // compute integral histogram
    void computeIntegralHistogram(int imageNumber);

    
    // generic sliding window  functions (for inference)
    void slidingWindow(Bbox& result, void (*slidingFunc)(Bbox&, double, short, short, short, short), bool rescale=true);
    
    // functions for sliding window
    static void slidingMax(Bbox& maxBbox, double score, short xl, short yl, short xh, short yh);
    static void slidingSum(Bbox& sumBbox, double score, short xl, short yl, short xh, short yh);

    // sliding window functions
    double slidingWindowLogSumExp(double *saveMaxScore=0); // computes log Z
    double slidingWindowLogSumExpCond(int var, const Bbox &bbox);

};  


/**
 * Calculate score of a box from integral image
 * A bounding box includes the (xl,yl,xh,yh) coordinates
 * computeBboxScore works directly in the quantized space
 *
 * Inspired by Christoph Lampert's Efficient Subwindow Search code
 * https://sites.google.com/a/christoph-lampert.com/work/software
 */ 
inline double ConditionalRandomField::computeBboxScore(short xl, short yl, short xh, short yh) {
  if ( (xl > xh) || (yl > yh) ) throw WRONG_BBOX;
  double val = integralImage[iiOffset(xh+1,yh+1)] - integralImage[iiOffset(xh+1,yl)]
             - integralImage[iiOffset(xl,yh+1)] + integralImage[iiOffset(xl,yl)];
  return val;
}

// compute feature map given any y
inline void ConditionalRandomField::computeFeatureMap(Ivector &featureMap, short xl, short yl, short xh, short yh) {
  if ( (xl > xh) || (yl > yh) ) throw WRONG_BBOX;
  for (size_t c=0; c<featureMap.size(); c++) {
    featureMap[c] = integralHistogram[iiOffset(xh+1,yh+1)][c] - integralHistogram[iiOffset(xh+1,yl)][c]
                  - integralHistogram[iiOffset(xl,yh+1)][c] + integralHistogram[iiOffset(xl,yl)][c];
  }
}

// convert (x,y) into 1d index
inline int ConditionalRandomField::iiOffset(int x, int y) {
  return y*iiWidth+x;
}


#endif // _CONDITIONAL_RANDOM_FIELD_H_

