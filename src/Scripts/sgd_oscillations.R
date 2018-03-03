fvals <- read.table("sgd_fvals_horse_8_5epochs.txt")
iterations = seq(1, length(fvals[,1]))

itsperepoch = 129

# plot epoch 1
epoch = 0
range = seq(epoch*itsperepoch, epoch*itsperepoch+itsperepoch)
pdf('sgd_epoch1.pdf')
plot(iterations[range], fvals[range,1], ylim=c(2000,10000), col="blue", xlab="iterations", ylab="log-likelihood")
dev.off()

# plot epoch 2
epoch = 1
range = seq(epoch*itsperepoch, epoch*itsperepoch+itsperepoch)
pdf('sgd_epoch2.pdf')
plot(iterations[range], fvals[range,1], ylim=c(2400,3000), col="blue", xlab="iterations", ylab="log-likelihood")
dev.off()

# plot epoch 3
epoch = 2
range = seq(epoch*itsperepoch, epoch*itsperepoch+itsperepoch)
pdf('sgd_epoch3.pdf')
plot(iterations[range], fvals[range,1], ylim=c(2300,2500), col="blue", xlab="iterations", ylab="log-likelihood")
dev.off()

# plot epoch 4
epoch = 3
range = seq(epoch*itsperepoch, epoch*itsperepoch+itsperepoch)
pdf('sgd_epoch4.pdf')
plot(iterations[range], fvals[range,1], ylim=c(2350,2400), col="blue", xlab="iterations", ylab="log-likelihood")
dev.off()

# plot epoch 5
epoch = 4
range = seq(epoch*itsperepoch, epoch*itsperepoch+itsperepoch)
pdf('sgd_epoch5.pdf')
plot(iterations[range], fvals[range,1], ylim=c(2350,2400), col="blue", xlab="iterations", ylab="log-likelihood")
dev.off()

# plot all epochs
range = seq(0, 5*itsperepoch)
pdf('sgd_epochs_all.pdf')
plot(iterations[range], fvals[range,1], ylim=c(2000,10000), col="blue", xlab="iterations", ylab="log-likelihood")
dev.off()
