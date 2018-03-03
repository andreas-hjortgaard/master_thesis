# model selection plot

# for tucow

# LBFGS
lambdas   <- c(0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000)
AUCtrain  <- c(0.895346, 0.895428, 0.895635, 0.894196, 0.901545, 0.906469, 0.875004, 0.733834, 0.642529)
AUCval    <- c(0.828808, 0.835998, 0.849786, 0.859584, 0.869711, 0.875843, 0.854145, 0.710556, 0.595825)

pdf("tucow_lbfgs_modelselection.pdf")
plot(lambdas, AUCtrain, type="o", col="red", log="x", ylim=c(0,1), xlab="lambda", ylab="AUC", pch=21)
lines(lambdas, AUCval, type="o", col="blue", pch=22)
title("LBFGS")
legend(5000,0.12, c("AUC training", "AUC validation"), col=c("red","blue"), pch=c(21,22), lty=c(1,1))
dev.off()


# SGD
lambdas   <- c(0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000)
AUCtrain  <- c(0.890396, 0.89536, 0.897915, 0.898964, 0.900932, 0.846463, 0.699486, 0.652726, 0.594277)
AUCval    <- c(0.842217, 0.845765, 0.863317, 0.8725, 0.872113, 0.826649, 0.655378, 0.604268, 0.512902)

pdf("tucow_sgd_modelselection.pdf")
plot(lambdas, AUCtrain, type="o", col="red", log="x", ylim=c(0,1), xlab="lambda", ylab="AUC", pch=21)
lines(lambdas, AUCval, type="o", col="blue", pch=22)
title("Stochastic gradient descent")
legend(5000,0.12, c("AUC training", "AUC validation"), col=c("red","blue"), pch=c(21,22), lty=c(1,1))
dev.off()



# CD
lambdas   <- c(0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000)
AUCtrain  <- c(0.891828, 0.896003, 0.897687, 0.906279, 0.905532, 0.861416, 0.691844, 0.601268, 0.608099)
AUCval    <- c(0.853812, 0.858128, 0.865075, 0.874638, 0.879485, 0.841168, 0.652627, 0.547119, 0.554748)

pdf("tucow_cd_modelselection.pdf")
plot(lambdas, AUCtrain, type="o", col="red", log="x", ylim=c(0,1), xlab="lambda", ylab="AUC", pch=21)
lines(lambdas, AUCval, type="o", col="blue", pch=22)
title("Contrastive divergence")
legend(5000,0.12, c("AUC training", "AUC validation"), col=c("red","blue"), pch=c(21,22), lty=c(1,1))
dev.off()



# PL
lambdas   <- c(0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000)
AUCtrain  <- c(0.898317, 0.894261, 0.894969, 0.889307, 0.904601, 0.908313, 0.890119, 0.805285, 0.647673)
AUCval    <- c(0.829183, 0.837663, 0.849082, 0.859535, 0.870931, 0.877732, 0.868893, 0.77945, 0.603724)

pdf("tucow_pl_modelselection.pdf")
plot(lambdas, AUCtrain, type="o", col="red", log="x", ylim=c(0,1), xlab="lambda", ylab="AUC", pch=21)
lines(lambdas, AUCval, type="o", col="blue", pch=22)
title("Pseudolikelihood")
legend(5000,0.12, c("AUC training", "AUC validation"), col=c("red","blue"), pch=c(21,22), lty=c(1,1))
dev.off()



# PW
lambdas   <- c(0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000)
AUCtrain  <- c(0.66641, 0.825243, 0.808954, 0.87247, 0.893, 0.891722, 0.884343, 0.812355, 0.473357)
AUCval    <- c(0.406974, 0.422855, 0.519103, 0.690317, 0.85298, 0.86546, 0.867012, 0.796391, 0.447072)

pdf("tucow_pw_modelselection.pdf")
plot(lambdas, AUCtrain, type="o", col="red", log="x", ylim=c(0,1), xlab="lambda", ylab="AUC", pch=21)
lines(lambdas, AUCval, type="o", col="blue", pch=22)
title("Piecewise training")
legend(5000,0.12, c("AUC training", "AUC validation"), col=c("red","blue"), pch=c(21,22), lty=c(1,1))
dev.off()




# for bicycles

# pseudolikelihood
#lambdas   <- c(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000)
#AUCtrain  <- c(0.566473, 0.56735, 0.458262, 0.466319, 0.500986, 0.516843, 0.622074, 0.716539, 0.68003, 0.573182, 0.437351, 0.423924)
#AUCval    <- c(0.404068, 0.403866, 0.398315, 0.393381, 0.382357, 0.378062, 0.395718, 0.462204, 0.513021, 0.533283, 0.461422, 0.449146)

#pdf("pseudo_modelselection.pdf")
#plot(lambdas, AUCtrain, type="o", col="red", log="x", ylim=c(0,1), xlab="lambda", ylab="AUC", pch=21)
#lines(lambdas, AUCval, type="o", col="blue", pch=22)
#title("PASCAL bicycle")
#legend(500,1, c("AUC training", "AUC validation"), col=c("red","blue"), pch=c(21,22), lty=c(1,1))
#dev.off()

# piecewise
#AUCtrain  <- c(0.62009, 0.625266, 0.625892, 0.61976, 0.625729, 0.62798, 0.652726, 0.668263, 0.645169, 0.610956, 0.515085, 0.428022)
#AUCval    <- c(0.330597, 0.32931, 0.329328, 0.33188, 0.330822, 0.336385, 0.341694, 0.383928, 0.451561, 0.528589, 0.518228, 0.451032)

#pdf("piecewise_modelselection.pdf")
#plot(lambdas, AUCtrain, type="o", col="red", log="x", ylim=c(0,1), xlab="lambda", ylab="AUC", pch=21)
#lines(lambdas, AUCval, type="o", col="blue", pch=22)
#title("PASCAL bicycle")
#legend(500,1, c("AUC training", "AUC validation"), col=c("red","blue"), pch=c(21,22), lty=c(1,1))
#dev.off()


# contrastive divergence
#AUCtrain  <- c(0.685826, 0.6232, 0.692043, 0.629041, 0.652463, 0.739418, 0.669797, 0.549181, 0.440143, 0.424426, 0.421061, 0.420797)
#AUCval    <- c(0.416638, 0.3924, 0.441494, 0.409174, 0.414926, 0.463681, 0.525356, 0.52912, 0.458951, 0.448981, 0.447161, 0.446315)

#pdf("cd_modelselection.pdf")
#plot(lambdas, AUCtrain, type="o", col="red", log="x", ylim=c(0,1), xlab="lambda", ylab="AUC", pch=21)
#lines(lambdas, AUCval, type="o", col="blue", pch=22)
#title("PASCAL bicycle")
#legend(500,1, c("AUC training", "AUC validation"), col=c("red","blue"), pch=c(21,22), lty=c(1,1))
#dev.off()
