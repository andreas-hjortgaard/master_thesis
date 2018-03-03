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

#include "ObjectiveFunction.h"


using namespace std;

// constructor
ObjectiveFunction::ObjectiveFunction(DataManager *dm, ConditionalRandomField *crfield, SearchIx si)
  : dataManager(dm), crf(crfield), searchIx(si), lambda(0.)
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
DataManager *ObjectiveFunction::getDataManager()  {
  return dataManager;
}
void ObjectiveFunction::setDataManager(DataManager *dm) {
  dataManager = dm;
}

ConditionalRandomField *ObjectiveFunction::getCRF() {
  return crf;
}

void ObjectiveFunction::setCRF(ConditionalRandomField *crfield) {
  crf = crfield;
}
    
SearchIx ObjectiveFunction::getSearchIx() {
  return searchIx;
}

void ObjectiveFunction::setSearchIx(SearchIx si) {
  searchIx = si;
}

void ObjectiveFunction::setSearchIx(int i) {
  searchIx.clear();
  searchIx.push_back(i);
}

// get and set lambda
double ObjectiveFunction::getLambda() {
  return lambda;
}

void ObjectiveFunction::setLambda(double lam) {
  lambda = lam;
}

int ObjectiveFunction::getNumImages() {
  return dataManager->getNumFiles();
}

SearchIx ObjectiveFunction::getNonEmpty() {
  return dataManager->getNonEmpty();
}

// functions from CRF
int ObjectiveFunction::getStepSize() {
  return crf->getStepSize();
}

IntegralImage *ObjectiveFunction::getIntegralImage() {
  return crf->getIntegralImage();
}

void ObjectiveFunction::computeIntegralImage(int imageNumber, Weights &w) {
  crf->computeIntegralImage(imageNumber, w);
  iiWidth = crf->getIntegralImageWidth();
  iiHeight = crf->getIntegralImageHeight();
}

double ObjectiveFunction::slidingWindowLogSumExp() {
  return crf->slidingWindowLogSumExp();
}
