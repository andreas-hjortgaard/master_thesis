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
#include <iostream>
#include <algorithm>
#include <limits>
#include <string>

#include "Types.h"
#include "DataManager.h"
#include "ConditionalRandomField.h"
#include "Inference/ESSWrapper.h"
#ifndef _GNUPLOT_
  #define _GNUPLOT_  
  #include "Lib/gnuplot-cpp/gnuplot_i.hpp"
#endif

using namespace std;

// area overlap for two boxes
double computeAreaOverlap(const Bbox& foundBbox, const Bbox& trueBbox) {
  
  // (left, top, right, bottom) of possibly overlapping region
  int left   = max(foundBbox.ltrb[LEFT],    trueBbox.ltrb[LEFT]);
  int top    = max(foundBbox.ltrb[TOP],     trueBbox.ltrb[TOP]);
  int right  = min(foundBbox.ltrb[RIGHT],   trueBbox.ltrb[RIGHT]);
  int bottom = min(foundBbox.ltrb[BOTTOM],  trueBbox.ltrb[BOTTOM]);

  // if no overlap, the area is zero
  if ((left > right) || (top > bottom)) return 0.;

  // compute the intersection
  // the "+1" is due to our convention that a bounding box is including its edges.
  int areaIntersection = (right - left + 1) * (bottom - top + 1);
  
  // compute the union
  int areaFound = (foundBbox.ltrb[RIGHT]-foundBbox.ltrb[LEFT]+1) * (foundBbox.ltrb[BOTTOM]-foundBbox.ltrb[TOP]+1);
  int areaTrue = (trueBbox.ltrb[RIGHT]-trueBbox.ltrb[LEFT]+1) * (trueBbox.ltrb[BOTTOM]-trueBbox.ltrb[TOP]+1);
  int areaUnion = areaFound + areaTrue - areaIntersection;
  
  // the area overlap is intersection/union
  return (double)areaIntersection / areaUnion;
}


// compute all area overlaps, used by average area overlap and recall overlap
Dvector computeAllAreaOverlaps(DataManager &dataman, SearchIx searchIx, int predictionStepSize, bool compareQuantized) {
  
  // if search index empty, create it
  if (searchIx.empty()) {
    searchIx = SearchIx(dataman.getNumFiles());
    for (size_t k=0; k<searchIx.size(); k++) {
      searchIx[k] = k;
    }
  }
  
  Images& image = dataman.getImages();
  Bboxes& bbox = dataman.getBboxes();
  Weights& weights = dataman.getWeights();
  
  int imageNumber;
  double areaOverlap;
  double tempAreaOverlap;
  Dvector areaOverlaps;
  for (size_t j=0; j<searchIx.size(); j++) {
    
    // extract image index
    imageNumber = searchIx[j];
    
    if (bbox[imageNumber].numObject == 0) continue;
    
    // find the maximum overlap in case of multiple boxes
    areaOverlap = 0.0;
    for (int i = 0; i < bbox[imageNumber].numObject; i++) {
    
      // compute score
      Bbox bestBbox;
      if (predictionStepSize == 1) { // compute using ESS, because it is faster
        bestBbox = computeESS(image[imageNumber], weights);
      } else { // compute using sliding window
        ConditionalRandomField crf(&dataman);
        crf.setStepSize(predictionStepSize);
        crf.computeIntegralImage(imageNumber, weights);
        bestBbox.ltrb = new short[4];
        bestBbox.score = -9999999.;
        crf.slidingWindow(bestBbox, ConditionalRandomField::slidingMax, !compareQuantized); // bestBbox is the _rescaled_ best bbox
      }
      
      // scale true bounding box if needed
      Bbox trueBbox;
      trueBbox.ltrb = new short[4];
      if (compareQuantized) {
        int scaledWidth  = image[imageNumber].width/predictionStepSize;
        int scaledHeight = image[imageNumber].height/predictionStepSize;
        trueBbox.ltrb[LEFT]   = min(bbox[imageNumber].ltrb[LEFT]/predictionStepSize, scaledWidth-1);
        trueBbox.ltrb[TOP]    = min(bbox[imageNumber].ltrb[TOP]/predictionStepSize, scaledHeight-1);
        trueBbox.ltrb[RIGHT]  = min(bbox[imageNumber].ltrb[RIGHT]/predictionStepSize, scaledWidth-1);
        trueBbox.ltrb[BOTTOM] = min(bbox[imageNumber].ltrb[BOTTOM]/predictionStepSize, scaledHeight-1);
      } else { // do not scale
        trueBbox.ltrb[LEFT]   = bbox[imageNumber].ltrb[LEFT];
        trueBbox.ltrb[TOP]    = bbox[imageNumber].ltrb[TOP];
        trueBbox.ltrb[RIGHT]  = bbox[imageNumber].ltrb[RIGHT];
        trueBbox.ltrb[BOTTOM] = bbox[imageNumber].ltrb[BOTTOM];
      }
      // compute Area Overlap with true bounding box
      tempAreaOverlap = computeAreaOverlap(bestBbox, trueBbox);
      delete[] bestBbox.ltrb;
      delete[] trueBbox.ltrb;
      
      // choose areaOverlap for true box with the largest overlap
      if (tempAreaOverlap > areaOverlap) areaOverlap = tempAreaOverlap;
    }   
    areaOverlaps.push_back(areaOverlap);
    
  }
  
  return areaOverlaps;
}


// compute averate area overlap for weights in dataman
double computeAverageAreaOverlap(DataManager &dataman, SearchIx searchIx, int predictionStepSize, bool compareQuantized) {
  
  // compute the area overlaps between the ground thruth and the predictions given the weights
  Dvector areaOverlaps = computeAllAreaOverlaps(dataman, searchIx, predictionStepSize, compareQuantized);
  
  // compute the sum of these overlaps
  double sumAreaOverlap = 0.;
  for (size_t i=0; i<areaOverlaps.size(); i++) {
    sumAreaOverlap += areaOverlaps[i];
  }
  
  // compute the average
  return sumAreaOverlap / areaOverlaps.size();
}


// compute the area under the recall-overlap curve using the trapezoidal rule
double computeRecallOverlapAUC(Dvector &overlap, Dvector &recall) {
  double AUC = 0.0;
  for (size_t i=1; i<overlap.size(); i++) {
    AUC += (overlap[i] - overlap[i-1]) * (recall[i-1] + recall[i]);
  }
  return AUC * 0.5;
}

// compute recall overlap
RecallOverlap computeRecallOverlap(DataManager &dataman, SearchIx searchIx, int predictionStepSize, bool compareQuantized) {
  int numPositives;
  RecallOverlap result;
  // compute the area overlaps between the ground thruth and the predictions given the weights
  result.overlap = computeAllAreaOverlaps(dataman, searchIx, predictionStepSize, compareQuantized);
  numPositives = result.overlap.size();
  result.recall.resize(numPositives);

  // sort the area overlaps in increasing order
  sort(result.overlap.begin(), result.overlap.end());
  
  // compute the recall value corresponding to each minimum overlap
  for (int i=0; i<numPositives; i++) {
    result.recall[i] = (double) (numPositives - i) / numPositives;
  }
  
  // insert begin and end values if needed
  // begin value of min overlap must be 0.0, as the curve starts at 1.0 here.
  if (result.overlap[0] > 0.0) {
    result.overlap.insert(result.overlap.begin(), 0.0);
    result.recall.insert(result.recall.begin(), 1.0);
  }
  // if end value of overlap is less than 1.0, we need to push back a new overlap 
  // value with the recall value 0.0, such that the area of the curve will be bounded here
  if (result.overlap[result.overlap.size()-1] < 1.0) {
    result.overlap.push_back(result.overlap[result.overlap.size()-1]);
    result.recall.push_back(0.0);
    
    // make dummy point to make xrange [0.0, 1.0]
    result.overlap.push_back(1.0);
    result.recall.push_back(0.0);
  }

  // compute the area under the recall-overlap curve
  result.AUC = computeRecallOverlapAUC(result.overlap, result.recall);
  
  return result;
}

// make recall overlap figure and store in a file
void printRecallOverlap(string plotName, RecallOverlap &recallOverlap) {

  // Draw a very nice plot with GNUPLOT!
  //string plotName = "test_output";
  try {
    Gnuplot g1("lines");
    cout << "Plotting recall-overlap curve" << endl;
    ostringstream os;
    os << "Recall-overlap curve\\n(AUC: " << recallOverlap.AUC << ")";
    string AUCString = os.str();
    g1.set_title(AUCString);
    g1.savetops(plotName);
    g1.set_grid();
    g1.set_xlabel("Minimum overlap").set_ylabel("Recall");
    g1.plot_xy(recallOverlap.overlap,recallOverlap.recall,"Recall-overlap");
  } 
  catch (GnuplotException ge) {
    cerr << ge.what() << endl;
  }

}


// make recall overlap figure and store in a file
void printRecallOverlap(string plotName, DataManager &dataman, int predictionStepSize, bool compareQuantized) {
  
  double averageAreaOverlap = computeAverageAreaOverlap(dataman, SearchIx(), predictionStepSize, compareQuantized);
  cout << "Average Area Overlap is: " << averageAreaOverlap << endl;
  
  RecallOverlap recallOverlap = computeRecallOverlap(dataman, SearchIx(), predictionStepSize, compareQuantized);
  cout << "Area under overlap-precision curve: " << recallOverlap.AUC << endl << endl;
   
  // Draw a very nice plot with GNUPLOT!
  //string plotName = "test_output";
  try {
    Gnuplot g1("lines");
    cout << "Plotting overlap-precision curve" << endl;
    ostringstream os;
    os << "Overlap-precision curve\\n(AUC: " << recallOverlap.AUC << ")";
    string AUCString = os.str();
    g1.set_title(AUCString);
    g1.savetops(plotName);
    g1.set_grid();
    g1.set_xlabel("Minimum overlap").set_ylabel("Precision");
    g1.plot_xy(recallOverlap.overlap,recallOverlap.recall,"Precision-overlap");
  } 
  catch (GnuplotException ge) {
    cerr << ge.what() << endl;
  }
}


// small routine to store recall-overlap in a file
// format:
// overlap length
// recall length
// AUC
// overlap[0]
// overlap[1]
// ..
// overlap[n]
// recall[0]
// recall[1]
// ..
// recall[m]
void storeRecallOverlap(std::string path, RecallOverlap recallOverlap) {

  ofstream recallOverlapFile(path.c_str());
  recallOverlapFile << recallOverlap.overlap.size() << endl;
  recallOverlapFile << recallOverlap.recall.size() << endl;
  recallOverlapFile << recallOverlap.AUC << std::endl;
  for (size_t i=0; i<recallOverlap.recall.size(); i++) {
    recallOverlapFile << recallOverlap.overlap[i] << "\n";
  }
  for (size_t i=0; i<recallOverlap.recall.size(); i++) {
    recallOverlapFile << recallOverlap.recall[i] << "\n";
  }
  recallOverlapFile.close();

}
