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

#ifndef _DATAMANAGER_H_
#define _DATAMANAGER_H_ 

#include <iostream>
#include <vector>
#include <string> 

#include "Types.h"

// class for holding extracted Image features, bounding boxes and weights
class DataManager {  

  private:
    // data: BOVW for each image
    // data[0][0]->x   : x coordinate of first image, first descriptor
    Images images;
    Bboxes bboxes;
    Weights weights;
    std::vector<std::string> filenames;
    int numFiles;
    
    // search index for images with object
    SearchIx nonEmpty;
    
    // clear and deallocate images and bboxes
    void clearImages();
    void clearBboxes();

  public:

    // constructors and desctructor
    DataManager();
    DataManager(std::string imgpath, std::string subsetpath);    
    DataManager(std::string imgpath, std::string bboxpath, std::string subsetpath);
    DataManager(std::string imgpath, std::string bboxpath, std::string subsetpath, std::string weightpath);
    
    ~DataManager();

    // getters/setters
    int getNumFiles();
    std::vector<std::string> &getFilenames();

    Images &getImages();
    void setImages(Images&);

    Bboxes &getBboxes();
    void setBboxes(Bboxes&);

    Weights &getWeights();
    void setWeights(Weights&);
    
    SearchIx getNonEmpty();
    
    // load images from a directory (binary)
    void loadImages(std::string path, std::string subset);

    // load images from a directory (ASCII)
    void loadImagesASCII(std::string path, std::string subset);

    // load annotations (optional, only use for training and validation sets)
    void loadBboxes(std::string path);

    // load weights from a directory (optional, use for predictions)
    void loadWeights(std::string path);

};

#endif // _DATAMANAGER_H_
