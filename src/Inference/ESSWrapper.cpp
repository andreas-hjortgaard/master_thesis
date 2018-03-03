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

#include "Types.h"
#include "quality_pyramid.hh"

Bbox computeESS(const Image& image, const Weights& weights) {
  // set up arguments for the ESS algorithm
  int argnumpoints = image.numFeatures;
  int argwidth = image.width;
  int argheight = image.height;
  double* argxpos = new double[argnumpoints];
  double* argypos = new double[argnumpoints];
  double* argclst = new double[argnumpoints];
  int argnumclusters = (int) weights.size();
  int argnumlevels = 1;
  double* argweight = new double[argnumclusters];
  for (int i = 0; i < argnumpoints; i++) {
    argxpos[i] = (double) image.x[i];
    argypos[i] = (double) image.y[i];
    argclst[i] = (double) image.c[i];
  }
  for (int i = 0; i < argnumclusters; i++) {
    argweight[i] = (double) weights[i];
  }
  
  // the ESS algorithm
  Box bestBox = pyramid_search(argnumpoints, argwidth, argheight,  argxpos, argypos, 
                               argclst, argnumclusters, argnumlevels, argweight);
  
  // clean up
  delete[] argxpos;
  delete[] argypos;
  delete[] argclst;
  delete[] argweight;
  
  // convert result to our Bbox type format
  Bbox result;
  result.numObject = 1;
  result.ltrb = new short[4];
  result.ltrb[0] = bestBox.left;
  result.ltrb[1] = bestBox.top;
  result.ltrb[2] = bestBox.right;
  result.ltrb[3] = bestBox.bottom;
  result.score = bestBox.score;
  
  return result;
}

