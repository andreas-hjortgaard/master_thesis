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

#ifndef _TYPES_H_
#define _TYPES_H_

#ifndef LINE_MAX
  #define LINE_MAX 2048
#endif
#define FILE_NOT_FOUND -1
#define ROUNDOFF_ERROR -2
#define DIM_ERROR -3
#define NOT_A_NUMBER -4
#define LAMBDA_TOO_SMALL -5
#define WRONG_BBOX -6
#define GRADIENT_SIZE_ERROR -7
#define STEP_SIZE_TOO_LARGE -8

// BFGS errors
#define LINESEARCH_ETA_TOO_SMALL -1000
#define LINESEARCH_MAX_STEP -999
#define LINESEARCH_POSITIVE_SLOPE -998
#define BFGS_ALREADY_MINIMIZED -997

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <sys/resource.h>


// enumeration of bounding box coordinates
const int LEFT    = 0;
const int TOP     = 1;
const int RIGHT   = 2;
const int BOTTOM  = 3;

// visual word tuple (x,y,c) represented as individual vectors
struct Image {  
  int height, width;  // height and width of the image
  int numFeatures;   // number of features
  short *x, *y, *c;   // descriptor position and visual word
};

// bounding box (object, left, top, right, bottom)
struct Bbox {
  int numObject;
  short *ltrb; //left, top, right, bottom;
  double score;
};

// the features type
typedef std::vector<Image> Images;

// annotations
typedef std::vector<Bbox> Bboxes;

// weights
typedef std::vector<double> Weights;

// ivector (int vector)
typedef std::vector<int> Ivector;

// dvector (double vector)
typedef std::vector<double> Dvector;

// dmatrix (double matrix)
typedef std::vector<Dvector> DMatrix;

// search indices
typedef std::vector<int> SearchIx;

// integral image
typedef std::vector<double> IntegralImage;

// integral histogram
typedef std::vector<Ivector> IntegralHistogram;

// recall overlap
struct RecallOverlap {
  double AUC;         // Area under the recall-overlap curve
  Dvector overlap;    // overlap values (1st axis)
  Dvector recall;     // recall values (2nd axis)
};


// small routine by Christoph Lampert
// compute the current time in seconds
inline double gettime() {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  struct timeval curTime = ru.ru_utime;
  double t = curTime.tv_sec*1.0 + (curTime.tv_usec/1000./1000.);
  return t; 
}


#endif // _TYPES_H_
