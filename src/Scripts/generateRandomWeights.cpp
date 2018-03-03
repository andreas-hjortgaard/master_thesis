#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>

using namespace std;

int main(int argc, char **argv) {

  if (argc < 3) {
    cout << "Please supply weight dimensionality and max value" << endl;
    return -1;
  }

  int weightDim     = atoi(argv[1]);
  double weightMax  = atof(argv[2]); 

  vector<double> randW(weightDim);

  cout << "Generating random weights of dim " << weightDim << "and max " << weightMax << endl;
  
  // start at some weights between -0.1 and 0.1
  srand(time(NULL)); rand();
  for (int i=0; i<weightDim; i++) {
    randW[i] = ((double) rand() / RAND_MAX)*(2*weightMax) - weightMax;
  }
  
  stringstream os;
  os << "randWeights_" << weightDim << "_" << weightMax << ".txt";
  string weightFile = os.str();
  
  ofstream weightFileStream(weightFile.c_str());
  
  if (!weightFileStream.is_open()) {
    cerr << "Could not open " << weightFile << endl;
  }
  
  cout << "Storing in file " << weightFile << endl;
  
  // store result in file
  for (int i=0; i<weightDim; i++) {
    weightFileStream << randW[i] << "\n";
  }
  weightFileStream.close();
  
  
  return 0;
 }
