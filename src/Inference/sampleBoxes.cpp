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

#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>

#include "DataManager.h"
#include "ConditionalRandomField.h"
#include "Inference/GibbsSampler.h"
#include "Types.h"

using namespace std;


void usage() {
  cout << "sampleBoxes [rootpath] [object] [weightPath] [imageNum] [numSteps] [numSamples] [stepSize]" << endl;
}

// script for loading an image and a set of weights and sample bboxes
int main(int argc, char **argv) {
  
  if (argc < 8) {
    usage();
    return 0;
  }
  
  string rootPath   = string(argv[1]);
  string object     = string(argv[2]);
  string weightPath = string(argv[3]);
  int imageNum      = atoi(argv[4]);
  int numSteps      = atoi(argv[5]);
  int numSamples    = atoi(argv[6]);
  int stepSize      = atoi(argv[7]);
  
  // setup CRF
  DataManager dataman;
  try {
    if (object.compare("tucow") == 0) {
      // if cow, use TUDarmstadt set
      cout << "Using TUDarmstadt cow set" << endl;
      dataman.loadImages(rootPath+"/cows-test/EUCSURF-3000/", rootPath+"/subsets/cows_test_width_height.txt");
      dataman.loadWeights(weightPath);
    } else {
      // else use PASCAL VOC dataset with different objects
      cout << "Using PASCAL " << object << endl;
      dataman.loadImages(rootPath+"/pascal/USURF3K/", rootPath+"/subsets/test_width_height.txt");
      dataman.loadWeights(weightPath); 
    }
  }
  catch (int e) {
    if (e == FILE_NOT_FOUND) {
      cerr << "There was a problem opening a file!" << endl;
      return -1;
    }
  }
  
  ConditionalRandomField crf(&dataman);

  // initialize Gibbs sampler
  Weights w = dataman.getWeights();
  crf.setStepSize(stepSize);                // set desired step size (one because we want perfect samples)
  crf.computeIntegralImage(imageNum, w);    // compute integral image for chosen image
  GibbsSampler gibbs(&crf);                 // create Gibbs sampler
    
  Bbox *sample;
  string filename = dataman.getFilenames()[imageNum];
  
  cout << "Sampled bounding boxes for image " << filename << endl;
  for (int i=0; i<numSamples; i++) {
    gibbs.initialize();                         // reinitialize with random bboxes
    gibbs.sample(numSteps, w, imageNum);        // run burn-in
    sample = gibbs.getCurrentSample(imageNum);
    
    // rescale
    sample->ltrb[LEFT]    = sample->ltrb[LEFT]*stepSize;
    sample->ltrb[TOP]     = sample->ltrb[TOP]*stepSize;
    sample->ltrb[RIGHT]   = (sample->ltrb[RIGHT]+1)*(stepSize-1);
    sample->ltrb[BOTTOM]  = (sample->ltrb[BOTTOM]+1)*(stepSize-1);
    
    cout << sample->ltrb[LEFT] << " " << sample->ltrb[TOP] << " " << sample->ltrb[RIGHT] << " " <<  sample->ltrb[BOTTOM] << " "; 
    cout.flush();
  }
  cout << endl;


  cout << "Done!" << endl;
  
  return 0;
}
