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

#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "DataManager.h"

using namespace std;

// implementation of the dataset class
DataManager::DataManager()
  : numFiles(0)
{
}

DataManager::DataManager(string imgpath, string subsetpath)
  : numFiles(0) 
{
  loadImages(imgpath, subsetpath);
}

DataManager::DataManager(string imgpath, string bboxpath, string subsetpath)
  : numFiles(0)
{
  loadImages(imgpath, subsetpath);
  loadBboxes(bboxpath);
}

DataManager::DataManager(string imgpath, string bboxpath, string subsetpath, string weightpath)
  : numFiles(0)
{
  loadImages(imgpath, subsetpath);
  loadBboxes(bboxpath);
  loadWeights(weightpath);
}

DataManager::~DataManager() {

  // free all allocated memory
  clearImages();
  clearBboxes();
  weights.clear();
}

void DataManager::clearImages() {
  
  // free all allocated descriptors
  for (size_t i = 0; i < images.size(); i++) {
    delete[] images[i].x;
    delete[] images[i].y;
    delete[] images[i].c;
  }
  
  images.clear();  
}

void DataManager::clearBboxes() {
  
  // free all bboxes
  for (size_t i = 0; i < bboxes.size(); i++) {
    delete[] bboxes[i].ltrb;
  }
  
  bboxes.clear();
}

int DataManager::getNumFiles() {
  return numFiles;
}

vector<string> &DataManager::getFilenames() {
  return filenames;
}

// get and set dataset
Images &DataManager::getImages() {
  return images;
}

void DataManager::setImages(Images &img) {
  images = img;
}

// get and set annotations
Bboxes &DataManager::getBboxes() {
  return bboxes;
}

void DataManager::setBboxes(Bboxes &bb) {
  bboxes = bb;
}

// get and set weights
Weights &DataManager::getWeights() {
  return weights;
}

void DataManager::setWeights(Weights &w) {
  weights = w;
}

// return a search index where number of objects > 0
SearchIx DataManager::getNonEmpty() {
  return nonEmpty;
}

// load dataset from directory (binary)
void DataManager::loadImages(string path, string subset) {

  // scan through subset list and and collect filenames and image dimensions
  ifstream fs, fsTest;
  istringstream is;
  char line[LINE_MAX];
  string filename;
  int width, height;
  vector<int> widths;
  vector<int> heights;

  fs.open(subset.c_str());
  if (!fs.is_open()) {
    throw FILE_NOT_FOUND;
  } 

  filenames.clear();  
  while (fs.getline(line, LINE_MAX)) {
    is.str(string(line));
    is >> filename >> width >> height;
    filenames.push_back(string(filename)+".xyc");
    widths.push_back(width);
    heights.push_back(height);
    is.clear();
  }
  fs.close();
  numFiles = filenames.size();

  // test if binary files exist
  string filepath;
  filepath = path + "/" + filenames[0];
  fs.open(filepath.c_str(), ios::binary);
  if (!fs.is_open()) {
    // if binary files does not exist, try ASCII version
    loadImagesASCII(path, subset);
    return;
  }

  fs.close();


  // process each file and extract (x,y,c) data
  // assumes to be stored as shorts as x_0 x_1 ... x_n y_0 y_1 ... y_n c_0 ...
  int filesize;
  int numFeatures;
  
  images.clear();
  images.resize(numFiles);
  for (int i = 0; i < numFiles; i++) {
  
    // compute the number of descriptors
    // size / (3*sizeof(short)) = size / (3*2)
    filepath = path + "/" + filenames[i];
    fs.open(filepath.c_str(), ios::binary);
    if (!fs.is_open()) {
      throw FILE_NOT_FOUND;
    }     

    // compute file sizeis.str(string(line));
    fs.seekg (0, ios::end);
    filesize = fs.tellg();
    fs.seekg (0, ios::beg);
    
    // read contents into arrays
    numFeatures = filesize/(3*sizeof(short));

    Image imageRep;
    imageRep.numFeatures = numFeatures;
    imageRep.width   = widths[i];
    imageRep.height  = heights[i];
    imageRep.x       = new short[numFeatures]; 
    imageRep.y       = new short[numFeatures];
    imageRep.c       = new short[numFeatures];
    

    fs.read((char *) imageRep.x, sizeof(short)*numFeatures);
    fs.read((char *) imageRep.y, sizeof(short)*numFeatures);
    fs.read((char *) imageRep.c, sizeof(short)*numFeatures);
    
    // store in dataset
    images[i] = imageRep;    

    fs.close();
  }

}

// load dataset from directory (binary)
void DataManager::loadImagesASCII(string path, string subset) {

  // scan through subset list and and collect filenames and image dimensions
  ifstream fs;
  istringstream is;
  char line[LINE_MAX];
  string filename;
  int width, height;
  vector<int> widths;
  vector<int> heights;

  fs.open(subset.c_str());
  if (!fs.is_open()) {
    throw FILE_NOT_FOUND;
  } 
  
  filenames.clear();
  while (fs.getline(line, LINE_MAX)) {
    is.str(string(line));
    is >> filename >> width >> height;
    filenames.push_back(string(filename)+".clst");
    widths.push_back(width);
    heights.push_back(height);
    is.clear();
  }
  fs.close();
  numFiles = filenames.size();

  // process each file and extract (x,y,c) data
  // assumes to be stored as shorts as x_0 x_1 ... x_n y_0 y_1 ... y_n c_0 ...
  int numFeatures;
  string filepath;

  clearImages();
  images.resize(numFiles);
  for (int i = 0; i < numFiles; i++) {
  
    // compute the number of descriptors
    // number of lines
    filepath = path + "/" + filenames[i];
    fs.open(filepath.c_str(), ios::in);
    if (!fs.is_open()) {
      throw FILE_NOT_FOUND;
    }     

    // compute number of features (equal to number of lines)
    numFeatures = 0;
    while (fs.getline(line, LINE_MAX)) {
      numFeatures++;
    }
    fs.close();
    
    // create image object
    Image imageRep;
    imageRep.numFeatures = numFeatures;
    imageRep.width   = widths[i];
    imageRep.height  = heights[i];
    imageRep.x       = new short[numFeatures]; 
    imageRep.y       = new short[numFeatures];
    imageRep.c       = new short[numFeatures];
    
    // read features
    fs.open(filepath.c_str(), ios::in);
    int j=0;
    double xx, yy, cc;    // if some features are double, convert to short TODO: ASK CHRISTOPH ABOUT DOUBLES
    while (fs.getline(line, LINE_MAX)) {      
      stringstream ss(line);
      ss >> xx >> yy >> cc;
      imageRep.x[j] = (short) xx;
      imageRep.y[j] = (short) yy;
      imageRep.c[j] = (short) cc;
      j++;
    }

    // store in dataset
    images[i] = imageRep;    

    fs.close();
  }

}


// load annotations from a single file (ASCII)
void DataManager::loadBboxes(string path) {
  
  ifstream fs;
  istringstream is;
  char line[LINE_MAX]; 
  string filename;
  vector<string> tokens;
  int numBboxs;
  
  fs.open(path.c_str());
  if (!fs.is_open()) {
    throw FILE_NOT_FOUND;
  }  
  
  // scan through file
  // insert (object, top, left, bottom right)
  clearBboxes();
  nonEmpty.clear();
  while (fs.getline(line, LINE_MAX)) {
    
    // parse line and add bboxs to annotations
    // if no bbox, add an empty box at (0,0,0,0)
    is.str(string(line));
    is >> filename >> numBboxs;
    
    Bbox bboxRep;
    bboxRep.numObject = numBboxs;

    // loads all bboxes into array
    if (numBboxs > 0) {
      bboxRep.ltrb = new short[numBboxs*4];
      for (int i = 0; i < numBboxs*4; i++) {
        is >> bboxRep.ltrb[i];
      }
    } else {
      bboxRep.ltrb = new short[4];
      bboxRep.ltrb[LEFT]    = 0;
      bboxRep.ltrb[TOP]     = 0;
      bboxRep.ltrb[RIGHT]   = 0;
      bboxRep.ltrb[BOTTOM]  = 0;
    }
    bboxes.push_back(bboxRep);
    is.clear();
  }
  fs.close();
  
  // add indices with object to nonEmpty
  for (size_t i=0; i<bboxes.size(); i++) {
    if (bboxes[i].numObject > 0) {
      nonEmpty.push_back((int)i);
    }
  }  
}


// load weights from a single file (ASCII)
void DataManager::loadWeights(string path) {

  ifstream fs;
  istringstream is;
  char line[LINE_MAX];
  double weight;

  fs.open(path.c_str());
  if (!fs.is_open()) {
    throw FILE_NOT_FOUND;
  }  

  // parse one weight per line
  while (fs.getline(line, LINE_MAX)) {
    is.str(string(line));    
    is >> weight;
    weights.push_back(weight);
    is.clear();
  }
  fs.close();

}

