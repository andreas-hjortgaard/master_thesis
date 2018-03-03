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

// implementation of the abstract objective function class
#include <cstdio>
#include <iostream>

#include "Gradient.h"


using namespace std;

// constructor
Gradient::Gradient(DataManager *dm, ConditionalRandomField *crfield, SearchIx si)
  : dataManager(dm), crf(crfield), searchIx(si)
{
  // if no search indices are defined, use the whole dataset
  if (dm != NULL && searchIx.empty()) {
    searchIx = SearchIx(dm->getNumFiles());
    for (size_t k=0; k<searchIx.size(); k++) {
      searchIx[k] = k;
    }
  } 
}

// getters/setters
DataManager *Gradient::getDataManager() {
  return dataManager;
}

void Gradient::setDataManager(DataManager *dm) {
  dataManager = dm;
}

ConditionalRandomField *Gradient::getCRF() {
  return crf;
}

void Gradient::setCRF(ConditionalRandomField *crfield) {
  crf = crfield;
}

    
SearchIx Gradient::getSearchIx() {
  return searchIx;
}

void Gradient::setSearchIx(SearchIx si) {
  searchIx = si;
}

void Gradient::setSearchIx(int i) {
  searchIx.clear();
  searchIx.push_back(i);
}

// get and set lambda
double Gradient::getLambda() {
  return lambda;
}

void Gradient::setLambda(double lam) {
  lambda = lam;
}

int Gradient::getNumImages() {
  return dataManager->getNumFiles();
}

// FUNCTIONS FROM CRF
int Gradient::getStepSize() {
  return crf->getStepSize();
}

IntegralHistogram *Gradient::getIntegralHistogram() {
  return crf->getIntegralHistogram();
}

// compute integral image
void Gradient::computeIntegralImage(int imageNumber, Weights &w) {
  crf->computeIntegralImage(imageNumber, w);
  iiWidth = crf->getIntegralImageWidth();
  iiHeight = crf->getIntegralImageHeight();
}

// compute integral histogram
void Gradient::computeIntegralHistogram(int imageNumber) {
  crf->computeIntegralHistogram(imageNumber);
  iiWidth = crf->getIntegralImageWidth();
  iiHeight = crf->getIntegralImageHeight();
}


double Gradient::slidingWindowLogSumExp() {
  return crf->slidingWindowLogSumExp();
}

