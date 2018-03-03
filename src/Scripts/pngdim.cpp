/* C++ program for extracting image dimensions from PNG files */ 

#include <iostream>
#include <fstream>
#include <arpa/inet.h>

using namespace std;

#define LINE_MAX 256

int main(int argc, char **argv) {
  
  if (argc < 3) {
    cerr << "Please supply PNG directory and subset file" << endl;
    return -1;
  }

  char *pngdir = argv[1];
  char *subset  = argv[2];

  // instream of filenames
  ifstream fs;
  char filename[LINE_MAX];
  
  fs.open(subset, fstream::in);
  if (!fs.is_open()) {
    cerr << "Could not open subset file" << endl;
    return -1;
  }

  // outstream to textfile
  fstream fout("filename_width_height.txt", fstream::out);
  if (!fout.is_open()) {
    cerr << "Could not open output file" << endl;
    return -1;
  }
  
  // run through files
  while (fs.getline(filename, LINE_MAX)) {
  
    // create file path, convert "2006_000001" to "000001.png"
    //string pngfile = string(pngdir) + "/" + string(filename).substr(5,6) + ".png";
    string pngfile = string(pngdir) + "/" + string(filename) + ".png";

    /* read PNG header 
     * Header layout: 
     * PNG signature (8 bytes)
     * Length of chunk (4 bytes)
     * Chunk type (4 bytes)
     * IHDR:
     * - Width (4 bytes)
     * - Height (4 bytes)
     * ...
     */ 
    unsigned int width, height;
    ifstream fin(pngfile.c_str(), fstream::in);
    if (!fin.is_open()) {
      cerr << "Could not open PNG file: " << pngfile << endl;
      return -1;
    }
     
    
    // seek to the width field which is 8+4+4 bytes into the file
    fin.seekg(16);
    fin.read((char *) &width, 4);
    fin.read((char *) &height, 4);
    fin.close();
    
    // change endianness
    width = ntohl(width);
    height = ntohl(height);
    
    fout << filename << " " << width << " " << height << endl;
    
    //cout << "Image " << argv[1] << " has width " << width << " and height " << height << endl;  
    //cout << "Image " << filename << endl;
    fin.clear();
    fin.close();  

  }
    
  fs.close();
  fout.close();
  
  return 0;
}
