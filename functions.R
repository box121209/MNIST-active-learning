
# sequence generator:
generate <- function(T, P){
  out <- c(1)
  for(t in 2:T){
    x <- sample(1:nclass,1, prob=P[out[t-1],])
    out <- c(out, x)
  }
  # return:
  out
}

baum.welch <- function(dat, P, 
                       variance=1.0,
                       update.covariance=TRUE,
                       niters=30,
                       verbose=TRUE){
  
  # rows of dat are the observed vector sequence
  # P is the known transition matrix
  
  # 'emission':
  y <- dat
  T <- nrow(dat) # nr time-steps
  k <- ncol(dat) # dimension
  
  # nr classes:
  nclass <- nrow(P)
  
  # initial q estimate:
  q <- rep(1/nclass,nclass)
  
  # initial mu estimate:
  mu0 <- lapply(1:nclass, function(i) rnorm(k))
  
  # initial sigma estimate:
  sigma0 <- lapply(1:nclass, function(i) diag(rep(variance, k)))
  
  # initialise alpha, beta vectors:
  alpha = matrix(
    nrow = nclass,
    ncol = T
  )
  beta = matrix(
    nrow = nclass,
    ncol = T
  )
  
  # intialise mu, sigma:
  mu <- mu0
  sigma <- sigma0
  
  for(iter in 1:niters){
    
    # E-step - forward-backward
    
    # prepare:
    inv_sig <- lapply(sigma, solve)
    ldet_sig <- lapply(sigma, function(mat) log(det(mat))/2)
    const <- k * log(2*pi)/2
    
    F <- function(v){
      # for vector v, returns density of v under each Gaussian,
      # one for each centroid and covariance
      f <- function(i){ 
        lg <- - t(v - mu[[i]]) %*% inv_sig[[i]] %*% (v - mu[[i]])/2 - ldet_sig[[i]] - const
        # return:
        exp(lg)
      }
      # return:
      sapply(1:nclass, f)
    }
    
    # alpha pass:
    alpha[,1] <- q * F(y[1,])
    for(j in 2:T){
      alpha[,j] <- ( t(alpha[,j-1]) %*% P ) * F(y[j,])
      alpha[,j] <- alpha[,j] / sum(alpha[,j])
    }
    
    # beta pass
    beta[,T] = rep(1,nclass)
    for(j in ((T-1):1)){
      beta[,j] <- t(beta[,j+1] * F(y[j+1,])) %*% t(P)
      beta[,j] <- beta[,j] / sum(beta[,j])  
    } 
    
    # gamma pass
    gamma <- alpha * beta
    zed <- colSums(gamma)
    for(i in 1:nclass) gamma[i,] <- gamma[i,] / zed
    
    # M-step - re-estimate q, mu, sigma
    q <- gamma[,1]
    
    mu <- as.list(data.frame( t(gamma %*% y) ))
    zed <- rowSums(gamma)
    for(j in 1:nclass) mu[[j]] <- mu[[j]] / zed[j]
    
    # note in passing that zed estimates the class sizes in the training data:
    if(verbose) cat(iter,":", sapply(zed, round), "\n")
    
    # covariance:
    if(update.covariance){
      sigma <- list()
      for(i in 1:nclass){
        tmp <- y - matrix(rep(mu[[i]], T), nrow=T, byrow=TRUE)
        sigma[[i]] <- t(tmp) %*% (gamma[i,] * tmp) / zed[i]
      } 
    }
  }
  
  # return:
  list(gamma=gamma,
       path = path <- sapply(1:T, function(t){ order(gamma[,t])[nclass] }),
       mu = mu,
       sigma = sigma)
}

