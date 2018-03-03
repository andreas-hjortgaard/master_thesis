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

#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "DataManager.h"
#include "ConditionalRandomField.h"
#include "Inference/GibbsSampler.h"
#include "Inference/ESSWrapper.h"
#include "Measures/LossMeasures.h"
#include "Types.h"

// INFERENCE TEST
using namespace std;
int main(int argc, char **argv) {

 // initialize data manager
  DataManager dataman;
    
  try {
    dataman.loadImages("../cows-train/EUCSURF-3000/", "../subsets/cows_train10_width_height.txt");
    dataman.loadBboxes("../cows-train/Annotations/TUcow_train10.ess");
    dataman.loadWeights("weights/sgd_weights_cows.txt");
  }
  catch (int e) {
    if (e == FILE_NOT_FOUND) {
      fprintf(stderr, "There was a problem opening a file!\n");
      return -1;
    }
  }
  
  // create log-likelihood objective function with regularization
  ConditionalRandomField crf(&dataman);
  Weights w = dataman.getWeights();
  //Weights w(3000, 0.0);  
  
  // set weights  
  int stepSize = 8;
  crf.setStepSize(stepSize);
  crf.setWeights(w);
  
  Bbox *sample;
  Dvector cHist;
  
  cout << "TESTING GIBBS SAMPLING!" << endl;
  // initialize Gibbs sampler 
  int numImages = crf.getNumImages();
  int numSteps = 20;
  int imageNum = 3;
  GibbsSampler gibbs(&crf);

  cout << "Initial bounding boxes:" << endl;
  for (int i=0; i<numImages; i++) {
    sample = gibbs.getCurrentSample(i);
    printf("%d %d %d %d\n", sample->ltrb[0], sample->ltrb[1], sample->ltrb[2], sample->ltrb[3]);
  }

  cout << "Sampling from image " << imageNum << endl;
  for (int i=0; i<numSteps; i++) {
    gibbs.sample(1, w, imageNum);
    sample = gibbs.getCurrentSample(imageNum);
    printf("%d %d %d %d\n", sample->ltrb[0], sample->ltrb[1], sample->ltrb[2], sample->ltrb[3]);
  }


  cout << endl << "TESTING ESS" << endl;
  Bbox bestbox = computeESS(dataman.getImages()[imageNum], w);
  printf("Best box for image %d: %d %d %d %d\n", imageNum, bestbox.ltrb[0], bestbox.ltrb[1], bestbox.ltrb[2], bestbox.ltrb[3]);


  cout << endl << "TESTING AVERAGE AREA OVERLAP" << endl;
  Bbox groundtruth; 
  Bbox sampleRescaled;
  sampleRescaled.ltrb = new short[4];
  
  // reinitialize Gibbs sampler
  gibbs.initialize();

  for (int i=0; i<numSteps; i++) {
    groundtruth = dataman.getBboxes()[imageNum];
    gibbs.sample(1, w, imageNum);
    sample = gibbs.getCurrentSample(imageNum); 
    sampleRescaled.ltrb[0] = sample->ltrb[0]*stepSize;
    sampleRescaled.ltrb[1] = sample->ltrb[1]*stepSize;
    sampleRescaled.ltrb[2] = (sample->ltrb[2]+1)*stepSize-1;
    sampleRescaled.ltrb[3] = (sample->ltrb[3]+1)*stepSize-1;
    printf("sample:       %d %d %d %d\n", sampleRescaled.ltrb[0], sampleRescaled.ltrb[1], sampleRescaled.ltrb[2], sampleRescaled.ltrb[3]);
    printf("ground truth: %d %d %d %d\n", groundtruth.ltrb[0], groundtruth.ltrb[1], groundtruth.ltrb[2], groundtruth.ltrb[3]);
    printf("area overlap: %.6f\n", computeAreaOverlap(groundtruth, sampleRescaled));
    printf("\n");
  }

  free(sampleRescaled.ltrb);

  cout << "Done!" << endl;
  
  return 0;
}
