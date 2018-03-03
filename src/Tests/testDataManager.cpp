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

#include "DataManager.h"

using namespace std;


int main(int argc, char **argv) {
    DataManager dataman;
    
    try {
      //dataman.loadImages("../pascal/USURF3K/", "../subsets/train_width_height.txt");
      //dataman.loadBboxes("../pascal/Annotations/ess/bicycle_train.ess");
      dataman.loadImagesASCII("../cows-test/EUCSURF-3000/", "../subsets/cows_test_width_height_sorted.txt");
      dataman.loadBboxes("../cows-test/Annotations/TUcow_test_sorted.ess");
      dataman.loadWeights("../pascal/Weights/bicycle2006-w.txt");
    }
    catch (int e) {
      if (e == FILE_NOT_FOUND) {
        fprintf(stderr, "There was a problem opening a file!\n");
        return -1;
      }
    }     
    
    int i = 0;
    Images image = dataman.getImages();
    fprintf(stdout, "Number of files: %d\n", dataman.getNumFiles());
    fprintf(stdout, "Number of features for file %d: %d\n", i, image[i].numFeatures);
    for (int i = 0; i < 5; i++) {
      fprintf(stdout, "image[%d] dim: %d x %d\n", i, image[i].width, image[i].height);
      for (int j = 0; j < 5; j++) {
        fprintf(stdout, "image[%d][%d] = (%d, %d, %d)\n", i, j, image[i].x[j], image[i].y[j], image[i].c[j]);
      }
    }        

    
    fprintf(stdout, "\n");

    Bboxes bbox = dataman.getBboxes();
    fprintf(stdout, "Number of annotations: %d\n", (int) bbox.size());
    for (int i = 0; i < 20; i++) {
      if (bbox[i].numObject > 1) {
        fprintf(stdout, "annot[%d] = %d", i, bbox[i].numObject);
        for (int numObj = 0; numObj < bbox[i].numObject; numObj++) {
          fprintf(stdout, ", (%d, %d, %d, %d)", bbox[i].ltrb[4*numObj+0], bbox[i].ltrb[4*numObj+1], bbox[i].ltrb[4*numObj+2], bbox[i].ltrb[4*numObj+3]);
        }
        fprintf(stdout, "\n");
      } else {
        fprintf(stdout, "annot[%d] = %d, (%d, %d, %d, %d)\n", i, bbox[i].numObject, bbox[i].ltrb[0], bbox[i].ltrb[1], bbox[i].ltrb[2], bbox[i].ltrb[3]);
      }
    }    
  
    fprintf(stdout, "\n");

    Weights weights = dataman.getWeights();
    fprintf(stdout, "Number of weights: %d\n", (int) weights.size());
    for (int i = 0; i < 10; i++) {
      fprintf(stdout, "weights[%d] = %.10f\n", i, weights[i]);
    }

    return 0;
}

