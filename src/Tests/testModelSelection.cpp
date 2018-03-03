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
#include <iostream>
#include <fstream>

#include "Types.h"
#include "DataManager.h"
#include "ObjectiveFunctions/LogLikelihood.h"
#include "ObjectiveFunctions/StochasticGradient.h"
#include "ObjectiveFunctions/SampledGradient.h"
#include "ObjectiveFunctions/PseudoLikelihood.h"
#include "ObjectiveFunctions/PseudoLikelihoodGradient.h"
#include "Learning/LBFGS.h"
#include "Learning/StochasticGradientDescent.h"
#include "Learning/ContrastiveDivergence.h"
#include "Measures/LossMeasures.h"
#include "ModelSelection/ModelSelection.h"
//#include "Lib/gnuplot-cpp/gnuplot_i.hpp"

using namespace std;


int main(int argc, char **argv) {

short stepSize = 16; // stepSize = 1 to use ESS. Otherwise use sliding window
//int computeWeights = 0;
int computeLossOfWeights = 0;

  // initialize data manager
  DataManager datamanTrain;
  DataManager datamanVal;
    
  try {
    datamanTrain.loadImages("../pascal/USURF3K/", "../subsets/train_width_height.txt");
    datamanTrain.loadBboxes("../pascal/Annotations/ess/bicycle_train.ess");
    datamanVal.loadImages("../pascal/USURF3K/", "../subsets/val_width_height.txt");
    datamanVal.loadBboxes("../pascal/Annotations/ess/bicycle_val.ess");
    //dataman.loadImages("../cows-train/EUCSURF-3000/", "subsets/cows_train_width_height.txt");
    //dataman.loadBboxes("../cows-train/Annotations/TUcow_train.ess");
  }
  catch (int e) {
    if (e == FILE_NOT_FOUND) {
      fprintf(stderr, "There was a problem opening a file!\n");
      return -1;
    }
  }
 
  ConditionalRandomField crf(&datamanTrain);
  crf.setStepSize(stepSize);
  
  // create log-likelihood objective function with regularization
  //PseudoLikelihood loglik(&datamanTrain, &crf);
  LogLikelihood loglik(&datamanTrain, &crf);  
  loglik.setLambda(0.0);//(2.0); no regularization
  
  //PseudoLikelihoodGradient loglikgrad(&datamanTrain, &crf);
  GibbsSampler gibbs(&crf);
  SampledGradient loglikgrad(&datamanTrain, &crf, &gibbs);  
  loglikgrad.setNumSamples(1);
  loglikgrad.setLambda(0.0);//(2.0); no regularization

  // set random weights
  int num_weights = 3000;
  Weights w(num_weights);
  
  // initialize weights between -1 and 1
  srand(time(NULL)); rand();
  for (int i=0; i<num_weights; i++) {
    w[i] = ((double) rand() / RAND_MAX)*2 - 1;
  }
  
  datamanTrain.setWeights(w);
  crf.setWeights(w);
  printf("%d\n", num_weights);
  //gibbs.initialize(w);  
  
  // test evaluate
  printf("StepSize: %d %d\n", loglik.getStepSize(), loglikgrad.getStepSize());    

  printf("Computing log-likelihood...\n");
  double fval   = loglik.evaluate(w);
  printf("Loglikelihood = %.6f\n", fval);

  printf("Computing gradient...\n");
  Dvector grad(num_weights);
  loglikgrad.evaluate(grad, w, 0);
  printf("Gradient = (%.2f, %.2f, %.2f, ...)\n", grad[0], grad[1], grad[2]);
  

  //LBFGS lbfgs(&loglik, &loglikgrad);
  //StochasticGradientDescent learner(&loglik, &loglikgrad);
  ContrastiveDivergence learner(&loglik, &loglikgrad);

  // MODEL SELECTION
  double lambda;  

  //lambda = modelSelection(datamanTrain, datamanVal, crf, loglik, loglikgrad, learner, -3, 7);
  lambda = modelSelectionStochastic(datamanTrain, datamanVal, crf, loglik, loglikgrad, learner, 3, 7);
  
  cout << "Best lambda: " << lambda << endl;
  
  /*ofstream weight_file("pseudo_likelihood_weights_cow.txt");
  if (!weight_file) {
    cerr << "Could not open file weights.txt" << endl;
  }
  
  // store result in file
  for (int i=0; i<num_weights; i++) {
    //printf("w[%d] = %.6f\n", (int) i, w_new[i]);
    weight_file << w_new[i] << endl;
  }

  weight_file.close();
  */
  cout << "Done!" << endl;
  
  //}
  
  if (computeLossOfWeights) {
    
    string pathImages = "../cows-train/EUCSURF-3000/";
    string pathSubset = "../subsets/cows_train_width_height.txt";
    string pathBboxes = "../cows-train/Annotations/TUcow_train.ess";
    //string pathWeights = "pseudo_likelihood_weights_cow.txt";
    string pathWeights = "cd_weights_bicycle.txt";
    
    DataManager lossDataMan(pathImages, pathBboxes, pathSubset, pathWeights);
    
    string plotName = "test_output";
    printRecallOverlap(plotName, lossDataMan, stepSize, false);
    /*//string pathImages = "../pascal/USURF3K/";
    //string pathSubset = "subsets/val_width_height.txt";
    //string pathBboxes = "../pascal/Annotations/ess/bicycle_val.ess";
    //string pathWeights = "pseudo_likelihood_weights_bicycle.txt";
    
    //string pathImages = "../pascal/USURF3K/";
    //string pathSubset = "subsets/train_width_height.txt";
    //string pathBboxes = "../pascal/Annotations/ess/bicycle_train.ess";
    //string pathWeights = "../pascal/Weights/bicycle2006-w.txt";
    //string pathWeights = "../pascal/Weights/bicycle2006-w.txt";
    
    DataManager lossDataMan(pathImages, pathBboxes, pathSubset, pathWeights);
    //srand(time(NULL)); rand();
    // initialize weights between -1 and 1
    //for (int i=0; i<num_weights; i++) {
    //  w[i] = ((double) rand() / RAND_MAX)*2 - 1;
    //}
    //lossDataMan.setWeights(w);
    
    double averageAreaOverlap = computeAverageAreaOverlap(lossDataMan, SearchIx(), stepSize);
    fprintf(stdout, "Average Area Overlap is: %.6f\n", averageAreaOverlap);
    
    RecallOverlap recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize);
    fprintf(stdout, "Area under recall-overlap curve: %.6f\n\n", recallOverlap.AUC);
    //fprintf(stdout, "Size of overlap vector: %d\n", (int)recallOverlap.overlap.size());
    //fprintf(stdout, "Size of recall vector: %d\n", (int)recallOverlap.recall.size());
    
    // Draw a very nice plot with GNUPLOT!
    string plotName = "test_output";
    try {
      Gnuplot g1("lines");
      cout << "*** plotting recall-overlap curve" << endl;
      ostringstream os;
      os << "Recall-overlap curve\\n(AUC: " << recallOverlap.AUC << ")";
      string AUCString = os.str();
      g1.set_title(AUCString);
      cout << endl << endl << "*** save to ps " << endl;
      g1.savetops(plotName);
      g1.set_grid();
      g1.set_xlabel("Minimum overlap").set_ylabel("Recall");
      g1.plot_xy(recallOverlap.overlap,recallOverlap.recall,"Recall-overlap");
    } 
    catch (GnuplotException ge) {
      cerr << ge.what() << endl;
    }*/

  }
  
  return 0;
}
