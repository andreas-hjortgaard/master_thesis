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

#ifndef _LOSSMEASURES_H_
#define _LOSSMEASURES_H_

#include <string>
#include "DataManager.h"
#include "Types.h"

// different loss measures

// area overlap for two boxes
double computeAreaOverlap(const Bbox& foundBbox, const Bbox& trueBbox);

// average area overlap for a dataset given weights
// choose whether the best box is computed in the quantized image (predictionStepSize) 
// and whether this predicted box should be compared to the quantized true 
// bounding box (compareQuantized).
double computeAverageAreaOverlap(DataManager &dataman, SearchIx searchIx, int predictionStepSize=1, bool compareQuantized=false);

// compute recall overlap for a dataset given weights
RecallOverlap computeRecallOverlap(DataManager &dataman, SearchIx searchIx, int predictionStepSize=1, bool compareQuantized=false);

// make recall overlap figure and store in a file
void printRecallOverlap(std::string plotName, RecallOverlap &recallOverlap);

// make recall overlap figure and store in a file
void printRecallOverlap(std::string plotName, DataManager &dataman, int predictionStepSize=1, bool compareQuantized=false);

// store recall overlap in a file...
void storeRecallOverlap(std::string path, RecallOverlap recallOverlap);

#endif // _LOSSMEASURES_H_
