# Conditional Random Fields for Object Localization
This is the master thesis written by Andreas Christian Bjergaarde Eilschou and Andreas Hjortgaard Danielsen in the spring of 2012. While the thesis was officially written at the [Department of Computer Science, University of Copenhagen (DIKU)](http://diku.dk/), the actual work was done at [IST Austria](https://ist.ac.at/) and was joinly supervised by Christian Igel (DIKU) and Christoph Lampert (IST Austria).

The code and the thesis is published here on Github for  purposes only and while the code will compile (see below), the programs will not run without the proper training data. Also the code is not further developed and maintained. All executable programs require the USURF features as well as the annotations for both the TU Darmstadt cow dataset and PASCAL VOC 2006 dataset. Please contact Christoph Lampert (chl@ist.ac.at) for how to obtain these.


## Contents of this directory:
- `src/`        - source code
- `subsets/`    - list of training, validation and test sets
- `weights/`    - learned weights


## Compilation
To compile the source code for the CRF programs do the following:

Compile ESS:
- go to src/Lib/ESS-1_1/
- run 'make clean'
- run 'make'

Compile libLBFGS:
- go to src/Lib/liblbfgs/
- run 'tar xvf liblbfgs-1.10.tar.gz'
- go to src/Lib/liblbfgs/Libliblbfgs-1.10/
- run './configure'
- run 'make'

Compile the CRF code:
- go to src
- run 'make clean'
- run 'make'
