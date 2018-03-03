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

// main program for computing the pseudo marginal of a corner (that is two connected sides) 
// of a bbox over a certain image given the weights 
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

#include "DataManager.h"
#include "ConditionalRandomField.h"
#include "Types.h"


using namespace std;


void usage() {
  cout << "cornerMarginalsPseudo [rootpath] [object] [train/val/test] [imageNumber] [method] [weightsPath]" << endl;
}

// chooses the objective, gradient and learning algorithm based on the input
int main(int argc, char **argv) {

  // ensure correct number of arguments
  if (argc < 7) {
    usage();  
    return -1;
  }

  // parse input
  string rootPath     = string(argv[1]);
  string object       = string(argv[2]);
  string partition    = string(argv[3]);
  int imageNumber     = atoi(argv[4]);
  string method       = string(argv[5]);
  string weightsPath  = string(argv[6]);
  
  int stepSize        = 1;

  // make lower case
  transform(object.begin(), object.end(), object.begin(), ::tolower);

  // create experimental setup
  DataManager dataman;
  
  try {
    if (object.compare("tucow") == 0) {
      // if cow, use TUDarmstadt set
      cout << "Using TUDarmstadt cow set" << endl;
      dataman.loadImages(rootPath+"/cows-train/EUCSURF-3000/", rootPath+"/subsets/cows_"+partition+"_width_height.txt");
      dataman.loadBboxes(rootPath+"/cows-train/Annotations/TUcow_"+partition+".ess");
    } else {
      // else use PASCAL VOC dataset with different objects
      cout << "Using PASCAL " << object << endl;
      string boxes  = rootPath+"/pascal/Annotations/ess/" + object + "_"+partition+".ess";
      dataman.loadImages(rootPath+"/pascal/USURF3K/", rootPath+"/subsets/"+partition+"_width_height.txt");
      dataman.loadBboxes(boxes);
    }
  }
  catch (int e) {
    if (e == FILE_NOT_FOUND) {
      cerr << "There was a problem opening a file!" << endl;
      return -1;
    }
  }
    
  // setup conditional random field
  ConditionalRandomField crf(&dataman);
  crf.setStepSize(stepSize);
  
  // load weights
  dataman.loadWeights(weightsPath);
  Weights weights = dataman.getWeights();
  crf.setWeights(weights);
  
  // select image
  crf.computeIntegralImage(imageNumber, weights);
  int iiWidth  = crf.getIntegralImageWidth();
  int iiHeight = crf.getIntegralImageHeight();
  
  // select bbox
  Bboxes &bboxes = dataman.getBboxes();
  Bbox &bbox = bboxes[imageNumber];
  
  // setup distribution files
  ostringstream os;
  os << rootPath << "/results/cornerMarginals/" << object << "_" << partition << "_imageNumber_" << imageNumber << "_" << method << "_stepSize_" << stepSize << "_width_" << iiWidth-1 << "_height_" << iiHeight-1 << "_pseudoLT.txt";
  string distFileLT = os.str();
  
  os.clear();
  os.str("");
  os << rootPath << "/results/cornerMarginals/" << object << "_" << partition << "_imageNumber_" << imageNumber << "_" << method << "_stepSize_" << stepSize << "_width_" << iiWidth-1 << "_height_" << iiHeight-1 << "_pseudoLB.txt";
  string distFileLB = os.str();
  
  os.clear();
  os.str("");
  os << rootPath << "/results/cornerMarginals/" << object << "_" << partition << "_imageNumber_" << imageNumber << "_" << method << "_stepSize_" << stepSize << "_width_" << iiWidth-1 << "_height_" << iiHeight-1 << "_pseudoRT.txt";
  string distFileRT = os.str();
  
  os.clear();
  os.str("");
  os << rootPath << "/results/cornerMarginals/" << object << "_" << partition << "_imageNumber_" << imageNumber << "_" << method << "_stepSize_" << stepSize << "_width_" << iiWidth-1 << "_height_" << iiHeight-1 << "_pseudoRB.txt";
  string distFileRB = os.str();
  

  // common variables
  Dvector logZ(4);
  Dvector cornerDist((iiWidth-1)*(iiHeight-1),0.0);
  double startComputeCorner, stopComputeCorner;
  double pCondL, pCondT, pCondR, pCondB;
  
  // store original bbox values
  int valL = bbox.ltrb[LEFT];
  int valT = bbox.ltrb[TOP];
  int valR = bbox.ltrb[RIGHT];
  int valB = bbox.ltrb[BOTTOM];
 
  // compute partition function
  cout << "Computing partition function..." << endl;
  double startComputeZ = gettime();
  
  logZ[LEFT] = crf.slidingWindowLogSumExpCond(LEFT, bbox);
  logZ[TOP] = crf.slidingWindowLogSumExpCond(TOP, bbox);
  logZ[RIGHT] = crf.slidingWindowLogSumExpCond(RIGHT, bbox);
  logZ[BOTTOM] = crf.slidingWindowLogSumExpCond(BOTTOM, bbox);
  
  double stopComputeZ = gettime();
  cout << "... in " << stopComputeZ-startComputeZ << " seconds" << endl;
 
  // compute distribution LT -- p(L | T,R,B) * p(T | L,R,B)
  cout << "Computing marginal distribution for corner LT..." << endl;
  startComputeCorner = gettime();
  
  for (int y = 0; y <= bbox.ltrb[BOTTOM]; y++) {
    bbox.ltrb[LEFT] = valL;                                                     // restore true L value
    bbox.ltrb[TOP] = y;                                                         // set new T value
    pCondT = crf.condP(TOP, bbox, imageNumber, weights, false, logZ[TOP]);      // compute p(T | L,R,B)
    bbox.ltrb[TOP] = valT;                                                      // restore true T value
    for (int x = 0; x <= bbox.ltrb[RIGHT]; x++) {
      bbox.ltrb[LEFT] = x;                                                      // set new L value
      pCondL = crf.condP(LEFT, bbox, imageNumber, weights, false, logZ[LEFT]);  // compute p(L | T,R,B)
      cornerDist[y*(iiWidth-1)+x] = pCondL*pCondT;
    }
  }
  bbox.ltrb[LEFT] = valL;                                                       // restore true L value
  bbox.ltrb[TOP] = valT;                                                        // restore true T value
  
  stopComputeCorner = gettime();
  cout << "... in " << stopComputeCorner-startComputeCorner << " seconds" << endl;

  // store distribution in file
  ofstream distFileLTStream(distFileLT.c_str());
  if (!distFileLTStream.is_open()) {
    cerr << "Could not open " << distFileLT << endl;
  }
  distFileLTStream.precision(30);
  for (int i=0; i<(iiWidth-1)*(iiHeight-1); i++) {
    distFileLTStream << fixed << cornerDist[i] << "\n";
  }
  distFileLTStream.close();
  cout << "Saved distribution in file: " << distFileLT << endl;
  
  // compute distribution LB
  cout << "Computing marginal distribution for corner LB..." << endl;
  startComputeCorner = gettime();
  cornerDist.clear();
  cornerDist.resize((iiWidth-1)*(iiHeight-1),0.0);
  
  for (int y = bbox.ltrb[TOP]; y < iiHeight-1; y++) {
    bbox.ltrb[LEFT] = valL;                                                     // restore true L value
    bbox.ltrb[BOTTOM] = y;                                                      // set new B value
    pCondB = crf.condP(BOTTOM, bbox, imageNumber, weights, false, logZ[BOTTOM]);// compute p(B | L,T,R)
    bbox.ltrb[BOTTOM] = valB;                                                   // restore true B value
    for (int x = 0; x <= bbox.ltrb[RIGHT]; x++) {
      bbox.ltrb[LEFT] = x;                                                      // set new L value
      pCondL = crf.condP(LEFT, bbox, imageNumber, weights, false, logZ[LEFT]);  // compute p(L | T,R,B)
      cornerDist[y*(iiWidth-1)+x] = pCondL*pCondB;
    }
  }
  bbox.ltrb[LEFT] = valL;                                                       // restore true L value
  bbox.ltrb[BOTTOM] = valB;                                                     // restore true T value
  
  stopComputeCorner = gettime();
  cout << "... in " << stopComputeCorner-startComputeCorner << " seconds" << endl;

  // store distribution in file
  ofstream distFileLBStream(distFileLB.c_str());
  if (!distFileLBStream.is_open()) {
    cerr << "Could not open " << distFileLB << endl;
  }
  distFileLBStream.precision(30);
  for (int i=0; i<(iiWidth-1)*(iiHeight-1); i++) {
    distFileLBStream << fixed << cornerDist[i] << "\n";
  }
  distFileLBStream.close();
  cout << "Saved distribution in file: " << distFileLB << endl;
  
  // compute distribution RT
  cout << "Computing marginal distribution for corner RT..." << endl;
  startComputeCorner = gettime();
  cornerDist.clear();
  cornerDist.resize((iiWidth-1)*(iiHeight-1),0.0);
  
  for (int y = 0; y <= bbox.ltrb[BOTTOM]; y++) {
    bbox.ltrb[RIGHT] = valR;                                                    // restore true R value
    bbox.ltrb[TOP] = y;                                                         // set new T value
    pCondT = crf.condP(TOP, bbox, imageNumber, weights, false, logZ[TOP]);      // compute p(T | L,R,B)
    bbox.ltrb[TOP] = valT;                                                      // restore true T value
    for (int x = bbox.ltrb[LEFT]; x < iiWidth-1; x++) {
      bbox.ltrb[RIGHT] = x;                                                     // set new R value
      pCondR = crf.condP(RIGHT, bbox, imageNumber, weights, false, logZ[RIGHT]);// compute p(R | L,T,B)
      cornerDist[y*(iiWidth-1)+x] = pCondR*pCondT;
    }
  }
  bbox.ltrb[RIGHT] = valL;                                                      // restore true R value
  bbox.ltrb[TOP] = valT;                                                        // restore true T value
  
  stopComputeCorner = gettime();
  cout << "... in " << stopComputeCorner-startComputeCorner << " seconds" << endl;

  // store distribution in file
  ofstream distFileRTStream(distFileRT.c_str());
  if (!distFileRTStream.is_open()) {
    cerr << "Could not open " << distFileRT << endl;
  }
  distFileRTStream.precision(30);
  for (int i=0; i<(iiWidth-1)*(iiHeight-1); i++) {
    distFileRTStream << fixed << cornerDist[i] << "\n";
  }
  distFileRTStream.close();
  cout << "Saved distribution in file: " << distFileRT << endl;
  
  // compute distribution RB
  cout << "Computing marginal distribution for corner RB..." << endl;
  startComputeCorner = gettime();
  cornerDist.clear();
  cornerDist.resize((iiWidth-1)*(iiHeight-1),0.0);

  for (int y = bbox.ltrb[TOP]; y < iiHeight-1; y++) {
    bbox.ltrb[RIGHT] = valR;                                                    // restore true R value
    bbox.ltrb[BOTTOM] = y;                                                      // set new B value
    pCondB = crf.condP(BOTTOM, bbox, imageNumber, weights, false, logZ[BOTTOM]);// compute p(B | L,T,R)
    bbox.ltrb[BOTTOM] = valB;                                                   // restore true B value
    for (int x = bbox.ltrb[LEFT]; x < iiWidth-1; x++) {
      bbox.ltrb[RIGHT] = x;                                                     // set new R value
      pCondR = crf.condP(RIGHT, bbox, imageNumber, weights, false, logZ[RIGHT]);// compute p(R | L,T,B)
      cornerDist[y*(iiWidth-1)+x] = pCondR*pCondB;
    }
  }
  bbox.ltrb[RIGHT] = valL;                                                      // restore true R value
  bbox.ltrb[BOTTOM] = valB;                                                     // restore true B value

  stopComputeCorner = gettime();
  cout << "... in " << stopComputeCorner-startComputeCorner << " seconds" << endl;

  // store distribution in file
  ofstream distFileRBStream(distFileRB.c_str());
  if (!distFileRBStream.is_open()) {
    cerr << "Could not open " << distFileRB << endl;
  }
  distFileRBStream.precision(30);
  for (int i=0; i<(iiWidth-1)*(iiHeight-1); i++) {
    distFileRBStream << fixed << cornerDist[i] << "\n";
  }
  distFileRBStream.close();
  cout << "Saved distribution in file: " << distFileRB << endl;
  
 
  return 0;
}

