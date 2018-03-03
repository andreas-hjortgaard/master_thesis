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

// main program for getting training error and validation error on a given dataset with a given set of weights
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

  // create train and validation setup
  DataManager datamanTrain;
  DataManager datamanVal;  

  try {
    if (object.compare("tucow") == 0) {
      cerr << "TUCOW dataset not configured correctly yet!" << endl;
      return -1;
      // if cow, use TUDarmstadt set
      cout << "Using TUDarmstadt cow set" << endl;
      datamanVal.loadImages(rootPath+"/cows-train/EUCSURF-3000/", rootPath+"/subsets/cows_test_width_height.txt");
      datamanVal.loadBboxes(rootPath+"/cows-train/Annotations/TUcow_test.ess");
      datamanVal.loadWeights(weightPath);
    } else {
      // else use PASCAL VOC dataset with different objects
      cout << "Using PASCAL " << object << endl;
      string boxesTrain = rootPath+"/pascal/Annotations/ess/" + object + "_train.ess";
      datamanTrain.loadImages(rootPath+"/pascal/USURF3K/", rootPath+"/subsets/train_width_height.txt");
      datamanTrain.loadBboxes(boxesTrain);
      datamanTrain.loadWeights(weightPath);
      string boxesVal   = rootPath+"/pascal/Annotations/ess/" + object + "_val.ess";
      datamanVal.loadImages(rootPath+"/pascal/USURF3K/", rootPath+"/subsets/val_width_height.txt");
      datamanVal.loadBboxes(boxesVal);
      datamanVal.loadWeights(weightPath); 
    }
  }
  catch (int e) {
    if (e == FILE_NOT_FOUND) {
      cerr << "There was a problem opening a file!" << endl;
      return -1;
    }
  }
  
  // setup train and val result files
  ostringstream os;
  os << rootPath << "/results/val/" << method << "_" << object << "_" << stepSize << "_" << lambda << "_valresults.txt";
  string valFile = os.str();
  
  os.clear();
  os.str("");
  os << rootPath << "/results/val/" << method << "_" << object << "_" << stepSize << "_" << lambda << "_recallOverlapTrain";
  string lossTrain = os.str();
  
  os.clear();
  os.str("");
  os << rootPath << "/results/val/" << method << "_" << object << "_" << stepSize << "_" << lambda << "_recallOverlapVal";
  string lossVal = os.str();
  
  cout << "valFile: " << valFile << endl;
  cout << "lossVal: " << lossVal << endl;
  
  // open output file
  ofstream infostream(valFile.c_str());
  infostream << "Method                  : " << method << endl;
  infostream << "Object                  : " << object << endl;
  infostream << "Lambda                  : " << lambda << endl << endl;
  
  
  // compute recall-overlap for training set and validation set
  cout << "Computing recall-overlap..." << endl;
  
  SearchIx indices;
  RecallOverlap recallOverlapTrain;
  recallOverlapTrain = computeRecallOverlap(datamanTrain, indices, 1, false);
  infostream << "AUC train               : " << recallOverlapTrain.AUC << endl;

  RecallOverlap recallOverlapVal;
  recallOverlapVal = computeRecallOverlap(datamanVal, indices, 1, false);
  infostream << "AUC val                 : " << recallOverlapVal.AUC << endl;  
  
  // create figures 
  printRecallOverlap(lossTrain, recallOverlapTrain);
  printRecallOverlap(lossVal, recallOverlapVal);
  
  // test quantized prediction step size and unscaled ground truth
  recallOverlapTrain = computeRecallOverlap(datamanTrain, indices, stepSize, false);
  infostream << "AUC train(step/unquan)  : " << recallOverlapTrain.AUC << endl;
  
  recallOverlapVal = computeRecallOverlap(datamanVal, indices, stepSize, false);
  infostream << "AUC val  (step/unquan)  : " << recallOverlapVal.AUC << endl;
  
  // test quantized prediction step size and scaled ground truth
  recallOverlapTrain = computeRecallOverlap(datamanTrain, indices, stepSize, true);
  infostream << "AUC train(step/quan)    : " << recallOverlapTrain.AUC << endl;
  
  recallOverlapVal = computeRecallOverlap(datamanVal, indices, stepSize, true);
  infostream << "AUC val  (step/quan)    : " << recallOverlapVal.AUC << endl;
  infostream.close(); 
  

}
