setwd("~/projects/201707-MNIST-active-learning")

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

plot(lambda$values[ord[1:20]], col='blue')

lambda$vectors[,ord[1]] # should be constant

vec <- lambda$vectors[,ord[2:10]] # our 9-dimensional embedding

# save
write.table(cbind(vec, zip.sym), 
            file="zip_fiedler.txt", 
            row.names=FALSE, 
            col.names=FALSE)

##############################################################
# check the 9-dimensional representation

dat <- read.table("zip_fiedler.txt")

library(Rtsne)

dims <- 2
tsne.output <- Rtsne(dat[,1:9], 
                     theta=0.4,
                     max_iter=1000,
                     dims=dims,
                     verbose=TRUE, 
                     check_duplicates=FALSE)

# eyeball:
if(dims==2) plot(tsne.output$Y, 
                 cex=0.3,
                 pch=sprintf("%d", dat[,10]),
                 col=dat[,10])



##############################################################
# symbol stepping model

nclass <- 10
a <- 0.5

b <- (1-a)/nclass
P <- matrix(
  nrow=nclass, ncol=nclass,
  byrow=TRUE,
  data = rep(b, nclass^2)
)
for(i in 1:(nclass-1)){ P[i,i+1]=a }; P[nclass,1]=a
q0 <- rep(1/nclass,nclass)


# E.g.

state <- generate(50, P)
plot(state, col='blue', type='b', pch=19)


##############################################################
# synthetic symbol sequence

T <- 1000

state <- generate(T,P)
centroids <- lapply(1:9, function(i){ v<-rep(0,9); v[i]<-1; v} )
centroids[[10]] <- rep(0,9)

sd <- 0.25
synth <- list()
for(i in 1:T) synth[[i]] <- centroids[[state[i]]] + rnorm(9,0,sd)
synth <- do.call(rbind, synth)

# eyeball
tsne.output <- Rtsne(synth, 
                     verbose=TRUE, 
                     check_duplicates=FALSE)

# eyeball:
plot(tsne.output$Y, cex=0.3, col=state)



##############################################################
# Baum-Welch decoding


# run BW on synthetic data:

dat <- synth
#bw <- baum.welch(dat, P, variance=0.0125, update.covariance=FALSE)
bw <- baum.welch(dat, P, variance=0.4, niters=100, update.covariance=TRUE)


##############################################################
# performance on synthetic data

# compare states:
table(bw$path, state)

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

# 9-dimensional:
zip <- read.table("zip_fiedler.txt")
zip.vec <- zip[,1:9]
zip.sym <- zip[,10]

T <- nrow(zip.vec)
k <- ncol(zip.vec)
state <- generate(T,P)

# prepare zip items in HMM order:
dat <- matrix(0,nrow=T, ncol=k)
idx <- lapply(1:10, function(i) which(zip.sym==i%%10))
for(i in 1:T){
  dat[i,] <- as.matrix(zip.vec[sample(idx[[state[i]]], 1),])
}
 
# run Baum-Welch: 
bw <- baum.welch(dat, P, variance=1.0, P, niters=100, update.covariance=TRUE)

# and check performance:
table(bw$path, state)

mosaicplot(t(bw$gamma[,1:50]))
confidence <- sapply(1:T, function(t) max(bw$gamma[,t]))
plot(-log(confidence), type='l', col='blue')

h <- hist(confidence, col='grey', breaks=20)

cond <- (confidence > 0.99)
table(bw$path[cond], state[cond])

# examine the low-confidence digits:


##############################################################
# turning this into a 'transductive' (?) classifier

idx <- lapply(1:nclass, function(i) which(state==i))
ssize <- 50

trainers <- lapply(1:nclass, function(i) sample(idx[[i]], ssize))
estimated <- lapply(trainers, function(s) table( sapply(s, function(j) bw$path[j]) ))
( mapping <- sapply(estimated, function(x) as.integer( row.names( x[rev(order(x))] )[1] )%%10 ) )
