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

// main program for running model selection on a given dataset with a given learning method
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

#include "DataManager.h"
#include "ConditionalRandomField.h"
#include "Types.h"

#include "Measures/LossMeasures.h"

using namespace std;


void usage() {
  cout << "testPerformance [rootpath] [object] [stepSize] [lambda] [method] [weightPath]" << endl;
}

// chooses the objective, gradient and learning algorithm based on the input
int main(int argc, char **argv) {

  // ensure correct number of arguments
  if (argc < 6) {
    usage();  
    return -1;
  }

  // parse input
  string rootPath     = string(argv[1]);
  string object       = string(argv[2]);
  int stepSize        = atoi(argv[3]);
  double lambda       = atof(argv[4]);
  string method       = string(argv[5]);
  string weightPath   = string(argv[6]);
  
  
  // make lower case
  transform(object.begin(), object.end(), object.begin(), ::tolower);

  // create test setup
  DataManager datamanTest;  

  try {
    if (object.compare("tucow") == 0) {
      // if cow, use TUDarmstadt set
      cout << "Using TUDarmstadt cow set" << endl;
      datamanTest.loadImages(rootPath+"/cows-test/EUCSURF-3000/", rootPath+"/subsets/cows_test_width_height.txt");
      datamanTest.loadBboxes(rootPath+"/cows-test/Annotations/TUcow_test.ess");
      datamanTest.loadWeights(weightPath);
    } else {
      // else use PASCAL VOC dataset with different objects
      cout << "Using PASCAL " << object << endl;
      string boxesTest  = rootPath+"/pascal/Annotations/ess/" + object + "_test.ess";
      datamanTest.loadImages(rootPath+"/pascal/USURF3K/", rootPath+"/subsets/test_width_height.txt");
      datamanTest.loadBboxes(boxesTest);
      datamanTest.loadWeights(weightPath); 
    }
  }
  catch (int e) {
    if (e == FILE_NOT_FOUND) {
      cerr << "There was a problem opening a file!" << endl;
      return -1;
    }
  }
  
  // setup test result files
  ostringstream os;
  os << rootPath << "/results/test/" << method << "_" << object << "_" << stepSize << "_" << lambda << "_testresults.txt";
  string testFile = os.str();
  
  os.clear();
  os.str("");
  os << rootPath << "/results/test/" << method << "_" << object << "_" << stepSize << "_" << lambda << "_recallOverlapTest";
  string lossTest = os.str();
  
  os.clear();
  os.str("");
  os << rootPath << "/results/test/" << method << "_" << object << "_" << stepSize << "_" << lambda << "_recallOverlapValues.txt";
  string recallOverlapFile = os.str();
  
  cout << "testFile: " << testFile << endl;
  cout << "lossTest: " << lossTest << endl;
  
  // open output file
  ofstream infostream(testFile.c_str());
  infostream << "Method                  : " << method << endl;
  infostream << "Object                  : " << object << endl;
  infostream << "Lambda                  : " << lambda << endl << endl;
  
  
  // compute recall-overlap for training set and validation set
  cout << "Computing recall-overlap..." << endl;
  
  SearchIx indices;
  RecallOverlap recallOverlapTest;
  recallOverlapTest = computeRecallOverlap(datamanTest, indices, 1, false);

  
  infostream << "AUC test                : " << recallOverlapTest.AUC << endl;
  
  // create figures 
  printRecallOverlap(lossTest, recallOverlapTest);
  
  // store recall-overlap
  storeRecallOverlap(recallOverlapFile, recallOverlapTest);
  
  // test quantized prediction step size and unscaled ground truth
  recallOverlapTest = computeRecallOverlap(datamanTest, indices, stepSize, false);
  infostream << "AUC test (step/unquan)  : " << recallOverlapTest.AUC << endl;
  
  // test quantized prediction step size and scaled ground truth
  recallOverlapTest = computeRecallOverlap(datamanTest, indices, stepSize, true);
  infostream << "AUC test (step/quan)    : " << recallOverlapTest.AUC << endl;
  infostream.close(); 
  

}
