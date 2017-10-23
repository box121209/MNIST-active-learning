setwd("~/projects/201707-MNIST-active-learning/MNIST-active-learning")

source("functions.R")

library(ElemStatLearn)

summary(zip.train)
zip.sym <- zip.train[,1]
zip.sym

zip.vec <- zip.train[,2:257]


##############################################################
# spectral dimensional reduction

dist.mat <- as.matrix(dist(zip.vec))
nclass <- nrow(dist.mat)
k <- 3

# create symmetric k-nearest adjacency matrix:
adj <- matrix(0, nrow=nclass, ncol=nclass)
for(i in 1:nclass){
  v <- dist.mat[i,]
  for(j in order(v)[2:(k+1)]) adj[i,j] <- 1
}
adj <- adj + t(adj)
adj <- (adj>0)

laplace <- diag(rowSums(adj)) - adj

lambda <- eigen(laplace, only.values=FALSE)

ord <- order(lambda$values)

plot(lambda$values[ord[1:nvalues]], col='blue')

lambda$vectors[,ord[1]] # should be constant

vec <- lambda$vectors[,ord[1 + 1:256]] # our <nvalues>-dimensional embedding

# save
write.table(cbind(vec, zip.sym), 
            file="zip_fiedler.txt", 
            row.names=FALSE, 
            col.names=FALSE)


##############################################################
# t-SNE dimensional reduction

library(Rtsne)

dims <- 2
tsne.output <- Rtsne(zip.vec, 
                     theta=0.4,
                     max_iter=1000,
                     dims=dims,
                     verbose=TRUE, 
                     check_duplicates=FALSE)

sne.vec <- tsne.output$Y

##############################################################
# check the low-dimensional representations

nvalues <- 9 # maximum 256
zf <- read.table("zip_fiedler.txt")
vec <- zf[,1:nvalues]
sym <- zf[,257]

dims <- 2
tsne.output <- Rtsne(vec, 
                     theta=0.4,
                     max_iter=1000,
                     dims=dims,
                     verbose=TRUE, 
                     check_duplicates=FALSE)

# eyeball:
if(dims==2) plot(tsne.output$Y, 
                 cex=0.3,
                 pch=sprintf("%d", sym),
                 col=sym)

# compare this with the original representation in pixel space:
tsne.output <- Rtsne(zip.vec, 
                     theta=0.4,
                     max_iter=1000,
                     dims=dims,
                     verbose=TRUE, 
                     check_duplicates=FALSE)
if(dims==2) plot(tsne.output$Y, 
                 cex=0.3,
                 pch=sprintf("%d", sym),
                 col=sym)

##############################################################
# symbol stepping model

nclass <- 10
a <- 0.75

b <- (1-a)/nclass
P <- matrix(
  nrow=nclass, ncol=nclass,
  byrow=TRUE,
  data = rep(b, nclass^2)
)
for(i in 1:(nclass-1)){ P[i,i+1]=a }; P[nclass,1]=a
q0 <- rep(1/nclass,nclass)


# E.g.

true_class <- generate(50, P)
plot(true_class, col='blue', type='b', pch=19)


##############################################################
# synthetic symbol sequence

T <- 1000

true_class <- generate(T,P)
centroids <- lapply(1:9, function(i){ v<-rep(0,9); v[i]<-1; v} )
centroids[[10]] <- rep(0,9)

sd <- 0.25
synth <- list()
for(i in 1:T) synth[[i]] <- centroids[[true_class[i]]] + rnorm(9,0,sd)
synth <- do.call(rbind, synth)

# eyeball
tsne.output <- Rtsne(synth, 
                     verbose=TRUE, 
                     check_duplicates=FALSE)

# eyeball:
plot(tsne.output$Y, cex=0.3, col=true_class)



##############################################################
# Baum-Welch decoding


# run BW on synthetic data:

dat <- synth
#bw <- baum.welch(dat, P, variance=0.0125, update.covariance=FALSE)
bw <- baum.welch(dat, P, variance=0.4, niters=100, update.covariance=TRUE)


##############################################################
# performance on synthetic data

# compare true_classs:
table(bw$path, true_class)

# compare centroids:
true <- do.call(rbind,centroids)
est <- t(as.matrix(data.frame(bw$mu)))

k <- 10
dist.mat <- matrix(0, nrow=k, ncol=k)
for(i in 1:k){
  for(j in 1:k)
    dist.mat[i,j] <- sqrt(sum((true[i,] - est[j,])^2))
}
image(dist.mat)

# compare covariances:
eig <- do.call(rbind, lapply(bw$sigma, function(m) eigen(m)$values))
eig

##############################################################
# run on Zip data

# 256-dimensional:
zip.vec <- zip.train[,2:257]
zip.sym <- zip.train[,1]

# 3-dimensional:
zip.vec <- sne.vec

# 9-dimensional:
zip <- read.table("zip_fiedler.txt")
zip.vec <- zip[,1:9]
zip.sym <- zip[, ncol(zip)]


T <- nrow(zip.vec)
k <- ncol(zip.vec)
true_class <- generate(T,P)

# prepare zip items in HMM order:
dat <- matrix(0,nrow=T, ncol=k)
idx <- lapply(1:10, function(i) which(zip.sym==i%%10))
for(i in 1:T){
  dat[i,] <- as.matrix(zip.vec[sample(idx[[true_class[i]]], 1),])
}
 
# run Baum-Welch: 
bw <- baum.welch(dat, P, variance=1.0, niters=30, update.covariance=TRUE)

##############################################################
# check performance - or skip to classifier construction below

table(bw$path, true_class)

mosaicplot(t(bw$gamma[,1:50]))
confidence <- sapply(1:T, function(t) max(bw$gamma[,t]))
plot(-log(confidence), type='l', col='blue')

h <- hist(confidence, col='grey', breaks=20)

cond <- (confidence > 0.99)
table(bw$path[cond], true_class[cond])

# how do we assess gamma?

entropy <- function(prob){
  p <- prob[prob > 0]
  - sum(p * log(p))
}

# first look at how the actual true_classs appear to gamma:
visibility <- function(s, plot=TRUE){
  prob <- rowSums(bw$gamma[,true_class==s])
  prob <- prob/sum(prob)
  if(plot)
    plot(prob, type='h', col='blue', lwd=5, frame.plot=0, 
         xlab="Estimated true_class", ylab="Score",
         main=sprintf("true_class %d", s))
  # return:
  list('estimate' = which(prob==max(prob)),
       'entropy' = entropy(prob))
}

sapply(1:10, visibility)

##############################################################
# turning this into a classifier

# store indices of the known (but hidden) classes in the sequence:
idx <- lapply(1:nclass, function(i) which(true_class==i))

# assume the class priors are known:
class_priors = sapply(1:nclass, function(i) length(idx[[i]])/ length(true_class))

# select class labels for training:
ssize <- 1
trainers <- lapply(1:nclass, function(i) sample(idx[[i]], ssize))

# Estimate from trainers
# P(digit class | HMM state) \propto P(HMM state | class) * P(class)
# Rows are digit classes, columns are HMM states:
class_given_HMMstate <- matrix(0, nrow = nclass, ncol = nclass)
for(digit_class in 1:nclass){
  state_probs <- bw$gamma[ , trainers[[digit_class]]]
  if(ssize > 1) 
    state_probs <- rowSums(state_probs)
  state_probs <- class_priors[digit_class] * state_probs
  class_given_HMMstate[,digit_class] <- state_probs/sum(state_probs)
}
# Note that column sums are 1

# We now use P(digit_class | pixel image) = 
#  \sum_{HMM states} P(class | state) * P(state | image)
# The first factor is the matrix just computed, and the second factor
# is the gamma matrix from BW:

class_probs <- solve(class_given_HMMstate) %*% bw$gamma
# UNEXPECTED: WHY THE INVERSION HERE?? BUT IT'S WHAT WORKS...

class_estimates <- sapply(1:length(true_class), 
                          function(i) which(class_probs[,i]==max(class_probs[,i])))

# condition of not being in the trainers list:
cond <- !((1:T) %in% do.call(c, trainers))

tab <- table(class_estimates[cond], true_class[cond])
print(tab)
cat("Success rate:", sum(diag(as.matrix(tab)))/length(true_class[cond]))

