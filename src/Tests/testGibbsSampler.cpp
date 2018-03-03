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
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>

#include "DataManager.h"
#include "ObjectiveFunctions/LogLikelihood.h"
#include "ObjectiveFunctions/LogLikelihoodGradient.h"
#include "ObjectiveFunctions/SampledGradient.h"
#include "Inference/GibbsSampler.h"
#include "Types.h"

using namespace std;

//int testGibbsSampler() {
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
  //Weights w(3000, 0.1);  

  // random weights
  /*Weights w(3000);
  srand(time(NULL));
  for (size_t i=0; i<w.size(); i++) {
    w[i] = ((double) rand()/RAND_MAX)*4-2;
    printf("%.6f\n", w[i]);
  } */ 

  // set weights  
  int stepSize = 20;
  crf.setStepSize(stepSize);
  crf.setWeights(w);
  
  cout << "TESTING GIBBS SAMPLING!" << endl;
  // initialize Gibbs sampler 
  int numSteps = 20;
  int imageNum = 3;
  GibbsSampler gibbs(&crf);
    
  Bbox *sample;
  Dvector cHist;
  double score;
  
  printf("Sampled bounding boxes for image %d:\n", imageNum);
  for (int i=0; i<numSteps; i++) {
    gibbs.sample(1, w, imageNum);    // run some steps
    sample = gibbs.getCurrentSample(imageNum);
    cHist = gibbs.getCumulativeHistogram();
    score = crf.computeBboxScore(sample->ltrb[0], sample->ltrb[1], sample->ltrb[2], sample->ltrb[3]);
    printf("Image %d: %d %d %d %d\t%.6f\n", imageNum, sample->ltrb[0], sample->ltrb[1], sample->ltrb[2], sample->ltrb[3], score);
    
    // print histogram if last is not 1.0
    //if (abs(cHist.back() - 1.0) > 1e-6) { 
    //  for (int i=0; i<cHist.size(); i++) {
    //    printf("%.6f ", cHist[i]);
    //  }
    //  printf("\n");
    //}
  }

  imageNum = 0;
  printf("Sampled bounding boxes for image %d:\n", imageNum);
  for (int i=0; i<numSteps; i++) {
    gibbs.sample(1, w, imageNum);    // run some steps
    sample = gibbs.getCurrentSample(imageNum);
    cHist = gibbs.getCumulativeHistogram();
    score = crf.computeBboxScore(sample->ltrb[0], sample->ltrb[1], sample->ltrb[2], sample->ltrb[3]);
    printf("Image %d: %d %d %d %d\t%.6f\n", imageNum, sample->ltrb[0], sample->ltrb[1], sample->ltrb[2], sample->ltrb[3], score);
    
    // print histogram if last is not 1.0
    //if (abs(cHist.back() - 1.0) > 1e-6) { 
    //  for (int i=0; i<cHist.size(); i++) {
    //    printf("%.6f ", cHist[i]);
    //  }
    //  printf("\n");
    //}
  }

  // compute computed sampled gradient and real gradient and compare
  LogLikelihood loglik(&dataman, &crf);
  loglik.setLambda(2.0);

  LogLikelihoodGradient loglikgrad(&dataman, &crf);
  loglikgrad.setLambda(2.0);

  SampledGradient samplgrad(&dataman, &crf, &gibbs);
  samplgrad.setLambda(2.0);
  samplgrad.setNumSamples(1000);  
  samplgrad.setSkip(100);

  Dvector gradReal(w.size());
  Dvector gradSample(w.size());
  
  
    
  printf("Computing real gradient...\n");
  loglikgrad.evaluate(gradReal, w);

  printf("Computing sampled gradient...\n");
  samplgrad.evaluate(gradSample, w);

  double snorm, rnorm;
  snorm = rnorm = 0;
  for (size_t i=0; i<gradReal.size(); i++) {
    snorm += gradSample[i]*gradSample[i];
    rnorm += gradReal[i]*gradReal[i];        
    if (abs(gradReal[i] - gradSample[i]) > 1.0) {
      printf("gradReal[%d] = %.3f  vs.  gradSampled[%d] = %.3f\n", (int)i, gradReal[i], (int)i, gradSample[i]);
    }
  }
  snorm = sqrt(snorm);
  rnorm = sqrt(rnorm);

  printf("Norms:\nSampled: %.6f\tReal: %.6f\n", snorm, rnorm);

  cout << "Done!" << endl;
  
  return 0;
}
