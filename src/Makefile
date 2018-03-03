CC    = g++

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CC 	+= -g -Wall -I.
else
	CC	+= -Wall -I.
endif

BIN_DIR			= Binaries
EXEC_DIR		= Exec

LIBLBFGS		= -ILib/liblbfgs/liblbfgs-1.10/include
LIBLBFGS_O	= Lib/liblbfgs/liblbfgs-1.10/lib/lbfgs.o 
ESS					= -ILib/ESS-1_1
ESS_O		 		= Lib/ESS-1_1/quality_pyramid.o Lib/ESS-1_1/quality_box.o Lib/ESS-1_1/ess.o

DATACRF_O		= $(BIN_DIR)/DataManager.o $(BIN_DIR)/ConditionalRandomField.o
LOSS_O			= $(BIN_DIR)/LossMeasures.o

OBJ_O				= $(BIN_DIR)/ObjectiveFunction.o $(BIN_DIR)/Gradient.o
LOGLIK_O		= $(BIN_DIR)/LogLikelihood.o $(BIN_DIR)/LogLikelihoodGradient.o
PSEUDO_O 		= $(BIN_DIR)/PseudoLikelihood.o $(BIN_DIR)/PseudoLikelihoodGradient.o
PIECE_O			= $(BIN_DIR)/PiecewiseConditionalRandomField.o $(BIN_DIR)/PiecewiseLogLikelihood.o $(BIN_DIR)/PiecewiseGradient.o
STOCH_O			= $(BIN_DIR)/StochasticGradient.o
SAMPLE_O		= $(BIN_DIR)/SampledGradient.o

INF_O		  	= $(BIN_DIR)/GibbsSampler.o $(BIN_DIR)/ESSWrapper.o
LEARN_O			= $(BIN_DIR)/GradientDescent.o 
LBFGS_O 		= $(BIN_DIR)/LBFGS.o
SGD_O				= $(BIN_DIR)/StochasticGradientDescent.o
CD_O 				= $(BIN_DIR)/ContrastiveDivergence.o

MODEL_O			= $(BIN_DIR)/ModelSelection.o

ALL_O 			= $(DATACRF_O) $(LOSS_O) $(OBJ_O) $(LOGLIK_O) $(PSEUDO_O) $(PIECE_O) 
ALL_O			 += $(STOCH_O) $(SAMPLE_O) $(INF_O) $(LEARN_O) $(LBFGS_O) $(SGD_O) $(CD_O) $(MODEL_O)

MPI_O       = $(BIN_DIR)/LogLikelihoodGradient_MPI.o $(BIN_DIR)/LBFGS_MPI.o


all: tests modelSelection cornerMarginals cornerMarginalsPseudo cornerMarginalsPiecewise factorMarginalsPiecewise
tests: $(ALL_O) testDataManager testInference testGibbsSampler testLearning testLBFGS testStochasticGradient testContrastiveDivergence testLogLikelihood testPseudoLikelihood testPiecewiseLogLikelihood testModelSelection testLossMeasures testLambda testRandomWeightLoss
modelSelection: modelSelectionLBFGS modelSelectionSGD modelSelectionCD modelSelectionPseudo modelSelectionPiecewise testPerformance valPerformance


# PROGRAMS
sampleBoxes: $(DATACRF_O) $(INF_O)
	$(CC) -o $(EXEC_DIR)/sampleBoxes $(ESS) $(ESS_O) $(DATACRF_O) $(INF_O) Programs/sampleBoxes.cpp


# MODEL SELECTIONS
modelSelectionLBFGS_MPI: $(ALL_O) $(MPI_O)
	mpic++ -Wall -I. -o $(EXEC_DIR)/modelSelectionLBFGS_MPI $(ESS) $(ESS_O) $(LIBLBFGS) $(LIBLBFGS_O) $(ALL_O) $(MPI_O) ModelSelection/modelSelectionLBFGS_MPI.cpp

modelSelectionLBFGS: $(ALL_O)
	$(CC) -o $(EXEC_DIR)/modelSelectionLBFGS $(ESS) $(ESS_O) $(LIBLBFGS) $(LIBLBFGS_O) $(ALL_O) ModelSelection/modelSelectionLBFGS.cpp

modelSelectionSGD: $(ALL_O)
	$(CC) -o $(EXEC_DIR)/modelSelectionSGD $(ESS) $(ESS_O) $(LIBLBFGS) $(LIBLBFGS_O) $(ALL_O) ModelSelection/modelSelectionSGD.cpp

modelSelectionCD: $(ALL_O)
	$(CC) -o $(EXEC_DIR)/modelSelectionCD $(ESS) $(ESS_O) $(LIBLBFGS) $(LIBLBFGS_O) $(ALL_O) ModelSelection/modelSelectionCD.cpp

modelSelectionPseudo: $(ALL_O)
	$(CC) -o $(EXEC_DIR)/modelSelectionPseudo $(ESS) $(ESS_O) $(LIBLBFGS) $(LIBLBFGS_O) $(ALL_O) ModelSelection/modelSelectionPseudo.cpp

modelSelectionPiecewise: $(ALL_O)
	$(CC) -o $(EXEC_DIR)/modelSelectionPiecewise $(ESS) $(ESS_O) $(LIBLBFGS) $(LIBLBFGS_O) $(ALL_O) ModelSelection/modelSelectionPiecewise.cpp

testPerformance: $(ALL_O)
	$(CC) -o $(EXEC_DIR)/testPerformance $(ESS) $(ESS_O) $(LIBLBFGS) $(LIBLBFGS_O) $(ALL_O) ModelSelection/testPerformance.cpp
	
valPerformance: $(ALL_O)
	$(CC) -o $(EXEC_DIR)/valPerformance $(ESS) $(ESS_O) $(LIBLBFGS) $(LIBLBFGS_O) $(ALL_O) ModelSelection/valPerformance.cpp


# CORNER MARGINALS
cornerMarginals: $(ALL_O)
	$(CC) -o $(EXEC_DIR)/cornerMarginals $(ESS) $(ESS_O) $(LIBLBFGS) $(LIBLBFGS_O) $(ALL_O) Inference/cornerMarginals.cpp
cornerMarginalsPseudo: $(ALL_O)
	$(CC) -o $(EXEC_DIR)/cornerMarginalsPseudo $(ESS) $(ESS_O) $(LIBLBFGS) $(LIBLBFGS_O) $(ALL_O) Inference/cornerMarginalsPseudo.cpp
cornerMarginalsPiecewise: $(ALL_O)
	$(CC) -o $(EXEC_DIR)/cornerMarginalsPiecewise $(ESS) $(ESS_O) $(LIBLBFGS) $(LIBLBFGS_O) $(ALL_O) Inference/cornerMarginalsPiecewise.cpp
factorMarginalsPiecewise: $(ALL_O)
	$(CC) -o $(EXEC_DIR)/factorMarginalsPiecewise $(ESS) $(ESS_O) $(LIBLBFGS) $(LIBLBFGS_O) $(ALL_O) Inference/factorMarginalsPiecewise.cpp

# GENERAL TESTS
testDataManager: $(DATACRF_O)
	$(CC) -o $(EXEC_DIR)/testDataManager $(DATACRF_O) Tests/testDataManager.cpp


# INFERENCE TESTS
testInference: $(DATACRF_O) $(INF_O) $(LOSS_O)
	$(CC) -o $(EXEC_DIR)/testInference $(ESS) $(ESS_O) $(DATACRF_O) $(INF_O) $(LOSS_O) Tests/testInference.cpp

testGibbsSampler: $(DATACRF_O) $(INF_O) $(OBJ_O) $(LOGLIK_O) $(STOCH_O) $(SAMPLE_O)
	$(CC) -o $(EXEC_DIR)/testGibbsSampler $(ESS) $(ESS_O) $(DATACRF_O) $(INF_O) $(OBJ_O) $(LOGLIK_O) $(STOCH_O) $(SAMPLE_O) Tests/testGibbsSampler.cpp


# LEARNING TESTS
testLearning: $(DATACRF_O) $(INF_O) $(OBJ_O) $(LOGLIK_O) $(PSEUDO_O) $(PIECE_O) $(STOCH_O) $(SAMPLE_O)
	$(CC) -o $(EXEC_DIR)/testLearning $(ESS) $(ESS_O) $(DATACRF_O) $(INF_O) $(OBJ_O) $(LOGLIK_O) $(PSEUDO_O) $(PIECE_O) $(STOCH_O) $(SAMPLE_O) Tests/testLearning.cpp

testLBFGS: $(DATACRF_O) $(OBJ_O) $(LOGLIK_O) $(LEARN_O) $(LBFGS_O)
	$(CC) -o $(EXEC_DIR)/testLBFGS $(LIBLBFGS) $(LIBLBFGS_O) $(DATACRF_O) $(OBJ_O) $(LOGLIK_O) $(LEARN_O) $(LBFGS_O) Tests/testLBFGS.cpp

testStochasticGradient: $(DATACRF_O) $(OBJ_O) $(LOGLIK_O) $(STOCH_O) $(LEARN_O) $(LBFGS_O) $(SGD_O)
	$(CC) -o $(EXEC_DIR)/testStochasticGradient $(LIBLBFGS) $(LIBLBFGS_O) $(DATACRF_O) $(OBJ_O) $(LOGLIK_O) $(STOCH_O) $(LEARN_O) $(LBFGS_O) $(SGD_O) Tests/testStochasticGradient.cpp

testContrastiveDivergence: $(DATACRF_O) $(OBJ_O) $(LOGLIK_O) $(STOCH_O) $(SAMPLE_O) $(INF_O) $(LEARN_O) $(LBFGS_O) $(SGD_O) $(CD_O)
	$(CC) -o $(EXEC_DIR)/testContrastiveDivergence $(ESS) $(ESS_O) $(LIBLBFGS) $(LIBLBFGS_O) $(DATACRF_O) $(OBJ_O) $(LOGLIK_O) $(STOCH_O) $(SAMPLE_O) $(INF_O) $(LEARN_O) $(LBFGS_O) $(SGD_O) $(CD_O) Tests/testContrastiveDivergence.cpp

testLogLikelihood: $(DATACRF_O) $(LOSS_O) $(OBJ_O) $(LOGLIK_O) $(LBFGS_O) $(LEARN_O) $(INF_O)
	$(CC) -o $(EXEC_DIR)/testLogLikelihood $(ESS) $(ESS_O) $(LIBLBFGS) $(LIBLBFGS_O) $(DATACRF_O) $(LOSS_O) $(OBJ_O) $(LOGLIK_O) $(LBFGS_O) $(LEARN_O) $(INF_O) Tests/testLogLikelihood.cpp

testPseudoLikelihood: $(DATACRF_O) $(LOSS_O) $(OBJ_O) $(PSEUDO_O) $(LBFGS_O) $(LEARN_O) $(INF_O)
	$(CC) -o $(EXEC_DIR)/testPseudoLikelihood $(ESS) $(ESS_O) $(LIBLBFGS) $(LIBLBFGS_O) $(DATACRF_O) $(LOSS_O) $(OBJ_O) $(PSEUDO_O) $(LBFGS_O) $(LEARN_O) $(INF_O) Tests/testPseudoLikelihood.cpp

testPiecewiseLogLikelihood: $(DATACRF_O) $(LOSS_O) $(OBJ_O) $(PIECE_O) $(LBFGS_O) $(LEARN_O) $(INF_O)
	$(CC) -o $(EXEC_DIR)/testPiecewiseLogLikelihood $(ESS) $(ESS_O) $(LIBLBFGS) $(LIBLBFGS_O) $(DATACRF_O) $(LOSS_O) $(OBJ_O) $(PIECE_O) $(LBFGS_O) $(LEARN_O) $(INF_O) Tests/testPiecewiseLogLikelihood.cpp

testModelSelection: $(ALL_O)
	$(CC) -o $(EXEC_DIR)/testModelSelection $(ESS) $(ESS_O) $(LIBLBFGS) $(LIBLBFGS_O) $(ALL_O) Tests/testModelSelection.cpp

testLossMeasures: $(DATACRF_O) $(INF_O) $(LOSS_O)
	$(CC) -o $(EXEC_DIR)/testLossMeasures $(ESS) $(ESS_O) $(DATACRF_O) $(INF_O) $(LOSS_O) Tests/testLossMeasures.cpp
	
testLambda: $(ALL_O)
	$(CC) -o $(EXEC_DIR)/testLambda $(ESS) $(ESS_O) $(LIBLBFGS) $(LIBLBFGS_O) $(ALL_O) Tests/testLambda.cpp
	
testRandomWeightLoss: $(DATACRF_O) $(INF_O) $(LOSS_O)
	$(CC) -o $(EXEC_DIR)/testRandomWeightLoss $(ESS) $(ESS_O) $(DATACRF_O) $(INF_O) $(LOSS_O) Tests/testRandomWeightLoss.cpp



# CONDITIONAL RANDOM FIELD AND DATA MANAGER
$(BIN_DIR)/DataManager.o:
	$(CC) -c DataManager.cpp -o $(BIN_DIR)/DataManager.o

$(BIN_DIR)/ConditionalRandomField.o:
	$(CC) -c ConditionalRandomField.cpp -o $(BIN_DIR)/ConditionalRandomField.o

$(BIN_DIR)/PiecewiseConditionalRandomField.o:
	$(CC) -c PiecewiseConditionalRandomField.cpp -o $(BIN_DIR)/PiecewiseConditionalRandomField.o

$(BIN_DIR)/ModelSelection.o:
	$(CC) -c ModelSelection/ModelSelection.cpp -o $(BIN_DIR)/ModelSelection.o


# MEASURES
$(BIN_DIR)/LossMeasures.o:
	$(CC) -c Measures/LossMeasures.cpp -o $(BIN_DIR)/LossMeasures.o
	
	
# INFERENCE
$(BIN_DIR)/ESSWrapper.o:
	$(CC) $(ESS) -c Inference/ESSWrapper.cpp -o $(BIN_DIR)/ESSWrapper.o
	
$(BIN_DIR)/GibbsSampler.o:
	$(CC) -c Inference/GibbsSampler.cpp -o $(BIN_DIR)/GibbsSampler.o


# OBJECTIVE FUNCTIONS AND GRADIENTS
$(BIN_DIR)/ObjectiveFunction.o:
	$(CC) -c ObjectiveFunctions/ObjectiveFunction.cpp -o $(BIN_DIR)/ObjectiveFunction.o

$(BIN_DIR)/LogLikelihood.o:
	$(CC) -c ObjectiveFunctions/LogLikelihood.cpp -o $(BIN_DIR)/LogLikelihood.o

$(BIN_DIR)/PseudoLikelihood.o:
	$(CC) -c ObjectiveFunctions/PseudoLikelihood.cpp -o $(BIN_DIR)/PseudoLikelihood.o

$(BIN_DIR)/PiecewiseLogLikelihood.o:
	$(CC) -c ObjectiveFunctions/PiecewiseLogLikelihood.cpp -o $(BIN_DIR)/PiecewiseLogLikelihood.o

$(BIN_DIR)/Gradient.o:
	$(CC) -c ObjectiveFunctions/Gradient.cpp -o $(BIN_DIR)/Gradient.o

$(BIN_DIR)/LogLikelihoodGradient.o:
	$(CC) -c ObjectiveFunctions/LogLikelihoodGradient.cpp -o $(BIN_DIR)/LogLikelihoodGradient.o
	
$(BIN_DIR)/LogLikelihoodGradient_MPI.o:
	mpic++ -Wall -I. -c ObjectiveFunctions/LogLikelihoodGradient_MPI.cpp -o $(BIN_DIR)/LogLikelihoodGradient_MPI.o

$(BIN_DIR)/PseudoLikelihoodGradient.o:
	$(CC) -c ObjectiveFunctions/PseudoLikelihoodGradient.cpp -o $(BIN_DIR)/PseudoLikelihoodGradient.o

$(BIN_DIR)/PiecewiseGradient.o:
	$(CC) -c ObjectiveFunctions/PiecewiseGradient.cpp -o $(BIN_DIR)/PiecewiseGradient.o

$(BIN_DIR)/StochasticGradient.o:
	$(CC) -c ObjectiveFunctions/StochasticGradient.cpp -o $(BIN_DIR)/StochasticGradient.o

$(BIN_DIR)/SampledGradient.o:
	$(CC) -c ObjectiveFunctions/SampledGradient.cpp -o $(BIN_DIR)/SampledGradient.o


# LEARNING
$(BIN_DIR)/GradientDescent.o:
	$(CC) -c Learning/GradientDescent.cpp -o $(BIN_DIR)/GradientDescent.o
	
$(BIN_DIR)/LBFGS.o:
	$(CC) $(LIBLBFGS) -c Learning/LBFGS.cpp -o $(BIN_DIR)/LBFGS.o
	
$(BIN_DIR)/LBFGS_MPI.o:
	mpic++ -Wall -I. $(LIBLBFGS) -c Learning/LBFGS_MPI.cpp -o $(BIN_DIR)/LBFGS_MPI.o
	
$(BIN_DIR)/StochasticGradientDescent.o:
	$(CC) -c Learning/StochasticGradientDescent.cpp -o $(BIN_DIR)/StochasticGradientDescent.o

$(BIN_DIR)/ContrastiveDivergence.o:
	$(CC) -c Learning/ContrastiveDivergence.cpp -o $(BIN_DIR)/ContrastiveDivergence.o



	
.PHONY: clean
clean:
	rm -f $(EXEC_DIR)/* $(BIN_DIR)/*.o
