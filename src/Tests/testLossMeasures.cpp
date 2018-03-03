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

// test of computing losses and printing figures
#include <iostream>
#include <string>

#include "Measures/LossMeasures.h"

using namespace std;


int main(int argc, char **argv) {

  string plotName, pathWeights;
  string pathImages = "../pascal/USURF3K/";
  string pathSubset = "../subsets/train_width_height.txt";
  string pathBboxes = "../pascal/Annotations/ess/bicycle_train.ess";
  int stepSize = 16;

  // FOR TRAINING SET
  
  

  // for lambda = 1e-6
  plotName = "weights/bicycle_contrastive_16/loss_train_lambda_1e-06";
  pathWeights = "weights/bicycle_contrastive_16/lambda_1e-06.txt";
  DataManager lossDataMan(pathImages, pathBboxes, pathSubset, pathWeights);
  RecallOverlap recallOverlap;
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 1e-5
  plotName = "weights/bicycle_contrastive_16/loss_train_lambda_1e-05";
  pathWeights = "weights/bicycle_contrastive_16/lambda_1e-05.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 0.0001
  plotName = "weights/bicycle_pseudolikelihood_16/loss_train_lambda_0.0001";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_1e-05.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 0.001
  plotName = "weights/bicycle_pseudolikelihood_16/loss_train_lambda_0.001";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_0.001.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 0.01
  plotName = "weights/bicycle_pseudolikelihood_16/loss_train_lambda_0.01";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_0.01.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 0.1
  plotName = "weights/bicycle_pseudolikelihood_16/loss_train_lambda_0.1";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_0.1.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 1
  plotName = "weights/bicycle_pseudolikelihood_16/loss_train_lambda_1";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_1.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 10
  plotName = "weights/bicycle_pseudolikelihood_16/loss_train_lambda_10";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_10.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 100
  plotName = "weights/bicycle_pseudolikelihood_16/loss_train_lambda_100";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_100.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 1000
  plotName = "weights/bicycle_pseudolikelihood_16/loss_train_lambda_1000";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_1000.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 10000
  plotName = "weights/bicycle_pseudolikelihood_16/loss_train_lambda_10000";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_10000.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 100000
  plotName = "weights/bicycle_pseudolikelihood_16/loss_train_lambda_100000";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_100000.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 1e+6
  plotName = "weights/bicycle_pseudolikelihood_16/loss_train_lambda_1e+06";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_1e+06.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 1e+7
  plotName = "weights/bicycle_pseudolikelihood_16/loss_train_lambda_1e+07";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_1e+07.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);
  


  
  // FOR VALIDATION SET
  pathSubset = "subsets/val_width_height.txt";
  pathBboxes = "../pascal/Annotations/ess/bicycle_val.ess";

  // for lambda = 1e-6
  plotName = "weights/bicycle_contrastive_16/loss_val_lambda_1e-06";
  pathWeights = "weights/bicycle_contrastive_16/lambda_1e-06.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 1e-5
  plotName = "weights/bicycle_contrastive_16/loss_val_lambda_1e-05";
  pathWeights = "weights/bicycle_contrastive_16/lambda_1e-05.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 0.0001
  plotName = "weights/bicycle_pseudolikelihood_16/loss_val_lambda_0.0001";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_1e-05.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 0.001
  plotName = "weights/bicycle_pseudolikelihood_16/loss_val_lambda_0.001";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_0.001.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 0.01
  plotName = "weights/bicycle_pseudolikelihood_16/loss_val_lambda_0.01";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_0.01.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 0.1
  plotName = "weights/bicycle_pseudolikelihood_16/loss_val_lambda_0.1";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_0.1.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 1
  plotName = "weights/bicycle_pseudolikelihood_16/loss_val_lambda_1";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_1.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 10
  plotName = "weights/bicycle_pseudolikelihood_16/loss_val_lambda_10";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_10.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 100
  plotName = "weights/bicycle_pseudolikelihood_16/loss_val_lambda_100";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_100.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 1000
  plotName = "weights/bicycle_pseudolikelihood_16/loss_val_lambda_1000";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_1000.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 10000
  plotName = "weights/bicycle_pseudolikelihood_16/loss_val_lambda_10000";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_10000.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 100000
  plotName = "weights/bicycle_pseudolikelihood_16/loss_val_lambda_100000";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_100000.txt";
  
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 1e+6
  plotName = "weights/bicycle_pseudolikelihood_16/loss_val_lambda_1e+06";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_1e+06.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  // for lambda = 1e+7
  plotName = "weights/bicycle_pseudolikelihood_16/loss_val_lambda_1e+07";
  pathWeights = "weights/bicycle_pseudolikelihood_16/lambda_1e+07.txt";
  lossDataMan.loadWeights(pathWeights);
  recallOverlap = computeRecallOverlap(lossDataMan, SearchIx(), stepSize, true);
  printRecallOverlap(plotName, recallOverlap);

  return 0;
}
