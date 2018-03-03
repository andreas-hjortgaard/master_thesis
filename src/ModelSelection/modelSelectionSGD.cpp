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

#include "ObjectiveFunctions/LogLikelihood.h"
#include "ObjectiveFunctions/StochasticGradient.h"

#include "Learning/StochasticGradientDescent.h"

#include "Measures/LossMeasures.h"

using namespace std;


void usage() {
  cout << "modelSelectionSGD [rootpath] [object] [stepSize] [lambda] [maxEpochs] [intialEta] [constantEta]" << endl;
  cout << "Options:" << endl;
  cout << "  maxEpochs        : number of epochs to run " << endl;
  cout << "  initialEta       : value of initial eta when testing " << endl;
  cout << "  constantEta      : use constant eta or not" << endl;
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
  int stepSize        = atoi(argv[3]);
  double lambda       = atof(argv[4]);
  int maxEpochs       = atoi(argv[5]);
  double initialEta   = atof(argv[6]);
  bool constantEta    = false;

  if (atoi(argv[7]) != 0) {
    cout << "Constant eta" << endl;
    constantEta = true;
  }
  
  // make lower case
  transform(object.begin(), object.end(), object.begin(), ::tolower);

  // create experimental setup
  DataManager datamanTrain;
  DataManager datamanVal;
  

  try {
    if (object.compare("tucow") == 0) {
      // if cow, use TUDarmstadt set
      cout << "Using TUDarmstadt cow set" << endl;
      datamanTrain.loadImages(rootPath+"/cows-train/EUCSURF-3000/", rootPath+"/subsets/cows_train_width_height.txt");
      datamanTrain.loadBboxes(rootPath+"/cows-train/Annotations/TUcow_train.ess");
      datamanVal.loadImages(rootPath+"/cows-train/EUCSURF-3000/", rootPath+"/subsets/cows_val_width_height.txt");
      datamanVal.loadBboxes(rootPath+"/cows-train/Annotations/TUcow_val.ess"); 
    } else {
      // else use PASCAL VOC dataset with different objects
      cout << "Using PASCAL " << object << endl;
      string boxesTrain  = rootPath+"/pascal/Annotations/ess/" + object + "_train.ess";
      string boxesVal    = rootPath+"/pascal/Annotations/ess/" + object + "_val.ess";
      datamanTrain.loadImages(rootPath+"/pascal/USURF3K/", rootPath+"/subsets/train_width_height.txt");
      datamanTrain.loadBboxes(boxesTrain);
      datamanVal.loadImages(rootPath+"/pascal/USURF3K/", rootPath+"/subsets/val_width_height.txt");
      datamanVal.loadBboxes(boxesVal);      
    }
  }
  catch (int e) {
    if (e == FILE_NOT_FOUND) {
      cerr << "There was a problem opening a file!" << endl;
      return -1;
    }
  }
    
  // create experimental setup
  int weightDim = 3000;
  ConditionalRandomField crf(&datamanTrain);
  crf.setStepSize(stepSize);
  LogLikelihood loglik(&datamanTrain, &crf);
  StochasticGradient loglikgrad(&datamanTrain, &crf);

  // set lambda and initial weights 
  loglik.setLambda(lambda);
  loglikgrad.setLambda(lambda);

  
  // setup log files and info file
  ostringstream os;
  os << rootPath << "/results/sgd/" << object << "_" << stepSize << "_" << lambda << "_" << maxEpochs << "_info.txt";
  string infoFile = os.str();
  
  os.clear();
  os.str("");
  os << rootPath << "/results/sgd/" << object << "_" << stepSize << "_" << lambda << "_" << maxEpochs << "_weights.txt";
  string weightFile = os.str();
  
  os.clear();
  os.str("");
  os << rootPath << "/results/sgd/" << object << "_" << stepSize << "_" << lambda << "_" << maxEpochs << "_tempWeights.txt";
  string tempWeightFile = os.str();
  
  // recall overlap figures
  os.clear();
  os.str("");
  os << rootPath << "/results/sgd/" << object << "_" << stepSize << "_" << lambda << "_" << maxEpochs << "_recallOverlapTrain";
  string lossTrain = os.str();
  
  os.clear();
  os.str("");
  os << rootPath << "/results/sgd/" << object << "_" << stepSize << "_" << lambda << "_" << maxEpochs << "_recallOverlapVal";
  string lossVal = os.str();

  // print info for model selection run
  ofstream infostream(infoFile.c_str());
  infostream << "Learning algorithm      : StochasticGradientDescent\n";
  infostream << "Objective function      : LogLikelihood\n";
  infostream << "Gradient                : StochasticGradient\n";
  infostream << "Object                  : " << object << "\n";
  infostream << "Step size               : " << stepSize << "\n";
  infostream << "Lambda                  : " << lambda << "\n";
  infostream << "Max epochs              : " << maxEpochs << "\n";
  infostream << "Initial eta             : " << initialEta << "\n";
  infostream << "Constant eta            : " << constantEta << "\n";
  infostream.close();
 

  // choose initial weights depending on whether to kickstart or not
  Weights initialW(weightDim);

  cout << "Using randomly initialized weights" << endl;
  // start at some weights between -0.1 and 0.1
  srand(time(NULL)); rand();
  for (int i=0; i<weightDim; i++) {
    initialW[i] = ((double) rand() / RAND_MAX)*0.2 - 0.1;
  }
  crf.setWeights(initialW);

  // create learning algorithm
  StochasticGradientDescent sgd(&loglik, &loglikgrad);
  sgd.setMaxEpochs(maxEpochs);
  sgd.setAlpha(lambda);
  sgd.setTempWeightsPath(tempWeightFile);

  // perform model selection
  double start, stop;

  Weights wNew;
  // initialize learning rate
  try {
    start = gettime();
    sgd.initializeLearningRate(initialW, initialEta, 0, true);
    wNew = sgd.learnWeights(initialW);
    stop = gettime();
  } 
  catch (int e) {
    cerr << "There was an error with error code " << e << endl;
    return e;
  }
    
    
  // compute recall-overlap for training set and validation set
  cout << "Computing recall-overlap..." << endl;
  datamanTrain.setWeights(wNew);
  SearchIx indices;
  RecallOverlap recallOverlapTrain;
  recallOverlapTrain = computeRecallOverlap(datamanTrain, indices, 1, false);

  datamanVal.setWeights(wNew);
  RecallOverlap recallOverlapVal;
  recallOverlapVal = computeRecallOverlap(datamanVal, indices, 1, false);
  
  double alpha  = sgd.getAlpha();
  double t0     = sgd.getT0();
  double eta    = 1.0/(alpha*t0);

  infostream.open(infoFile.c_str(), ios_base::app);
  infostream << "alpha                   : " << alpha << endl;
  infostream << "t0                      : " << t0 << endl;
  infostream << "eta                     : " << eta << endl;
  infostream << "Time taken              : " << stop-start << endl << endl;
  infostream << "AUC train               : " << recallOverlapTrain.AUC << endl;
  infostream << "AUC val                 : " << recallOverlapVal.AUC << endl << endl;
  
  // create figures 
  printRecallOverlap(lossTrain, recallOverlapTrain);
  printRecallOverlap(lossVal, recallOverlapVal);
  
  // test quantized prediction step size and unscaled ground truth
  recallOverlapTrain = computeRecallOverlap(datamanTrain, indices, stepSize, false);
  recallOverlapVal = computeRecallOverlap(datamanVal, indices, stepSize, false);
  infostream << "AUC train (step/unquan) : " << recallOverlapTrain.AUC << endl;
  infostream << "AUC val   (step/unquan) : " << recallOverlapVal.AUC << endl << endl;
  
  // test quantized prediction step size and scaled ground truth
  recallOverlapTrain = computeRecallOverlap(datamanTrain, indices, stepSize, true);
  recallOverlapVal = computeRecallOverlap(datamanVal, indices, stepSize, true);
  infostream << "AUC train (step/quan)   : " << recallOverlapTrain.AUC << endl;
  infostream << "AUC val   (step/quan)   : " << recallOverlapVal.AUC << endl;
  infostream.close(); 
  
  // store weights
  // save learned weights in file
  ofstream weightFileStream(weightFile.c_str());
  if (!weightFileStream.is_open()) {
    cerr << "Could not open " << weightFile << endl;
  }
  
  // store result in file
  for (int i=0; i<weightDim; i++) {
    weightFileStream << wNew[i] << "\n";
  }
  weightFileStream.close();
  
  cout << "Done!" << endl;  

  return 0;
}

