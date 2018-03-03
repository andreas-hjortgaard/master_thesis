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

// main program for computing the marginal of a corner (that is two connected sides) 
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
  cout << "cornerMarginals [rootpath] [object] [train/val/test] [imageNumber] [method] [stepSize] [weightsPath]" << endl;
}

// chooses the objective, gradient and learning algorithm based on the input
int main(int argc, char **argv) {

  // ensure correct number of arguments
  if (argc < 8) {
    usage();  
    return -1;
  }

  // parse input
  string rootPath     = string(argv[1]);
  string object       = string(argv[2]);
  string partition    = string(argv[3]);
  int imageNumber     = atoi(argv[4]);
  string method       = string(argv[5]);
  int stepSize        = atoi(argv[6]);
  string weightsPath  = string(argv[7]);

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
  
  // load image
  crf.computeIntegralImage(imageNumber, weights);
  int iiWidth  = crf.getIntegralImageWidth();
  int iiHeight = crf.getIntegralImageHeight();
  
  // setup distribution files
  ostringstream os;
  os << rootPath << "/results/cornerMarginals/" << object << "_" << partition << "_imageNumber_" << imageNumber << "_" << method << "_stepSize_" << stepSize << "_width_" << iiWidth-1 << "_height_" << iiHeight-1 << "_LT.txt";
  string distFileLT = os.str();
  
  os.clear();
  os.str("");
  os << rootPath << "/results/cornerMarginals/" << object << "_" << partition << "_imageNumber_" << imageNumber << "_" << method << "_stepSize_" << stepSize << "_width_" << iiWidth-1 << "_height_" << iiHeight-1 << "_LB.txt";
  string distFileLB = os.str();
  
  os.clear();
  os.str("");
  os << rootPath << "/results/cornerMarginals/" << object << "_" << partition << "_imageNumber_" << imageNumber << "_" << method << "_stepSize_" << stepSize << "_width_" << iiWidth-1 << "_height_" << iiHeight-1 << "_RT.txt";
  string distFileRT = os.str();
  
  os.clear();
  os.str("");
  os << rootPath << "/results/cornerMarginals/" << object << "_" << partition << "_imageNumber_" << imageNumber << "_" << method << "_stepSize_" << stepSize << "_width_" << iiWidth-1 << "_height_" << iiHeight-1 << "_RB.txt";
  string distFileRB = os.str();
  

  // common variables
  double logZ, maxScore;
  Bbox bbox;
  bbox.ltrb = new short[4];
  Dvector cornerDist((iiWidth-1)*(iiHeight-1));
  double startComputeCorner, stopComputeCorner;
 
  // compute partition function
  cout << "Computing partition function..." << endl;
  double startComputeZ = gettime();
  logZ = crf.slidingWindowLogSumExp(&maxScore);
  double stopComputeZ = gettime();
  cout << "... in " << stopComputeZ-startComputeZ << " seconds" << endl;
 
  // compute distribution LT
  cout << "Computing marginal distribution for corner LT..." << endl;
  startComputeCorner = gettime();
  for (int y = 0; y < iiHeight-1; y++) {
    bbox.ltrb[TOP] = y;
    for (int x = 0; x < iiWidth-1; x++) {
      bbox.ltrb[LEFT] = x;
      cornerDist[y*(iiWidth-1)+x] = crf.cornerP(LEFT, TOP, bbox, imageNumber, weights, false, logZ, maxScore);
    }
  }
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
  for (int y = 0; y < iiHeight-1; y++) {
    bbox.ltrb[BOTTOM] = y;
    for (int x = 0; x < iiWidth-1; x++) {
      bbox.ltrb[LEFT] = x;
      cornerDist[y*(iiWidth-1)+x] = crf.cornerP(LEFT, BOTTOM, bbox, imageNumber, weights, false, logZ, maxScore);
    }
  }
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
  for (int y = 0; y < iiHeight-1; y++) {
    bbox.ltrb[TOP] = y;
    for (int x = 0; x < iiWidth-1; x++) {
      bbox.ltrb[RIGHT] = x;
      cornerDist[y*(iiWidth-1)+x] = crf.cornerP(RIGHT, TOP, bbox, imageNumber, weights, false, logZ, maxScore);
    }
  }
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
  for (int y = 0; y < iiHeight-1; y++) {
    bbox.ltrb[BOTTOM] = y;
    for (int x = 0; x < iiWidth-1; x++) {
      bbox.ltrb[RIGHT] = x;
      cornerDist[y*(iiWidth-1)+x] = crf.cornerP(RIGHT, BOTTOM, bbox, imageNumber, weights, false, logZ, maxScore);
    }
  }
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


  delete[] bbox.ltrb;

  return 0;
}

