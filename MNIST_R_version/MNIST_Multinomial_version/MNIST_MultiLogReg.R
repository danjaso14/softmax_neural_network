

suppressMessages(library(reticulate))
suppressMessages(library(tensorflow))
suppressMessages(library(keras))



mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y


# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# rescale
x_train <- x_train / 255
x_test <- x_test / 255


# The to_categorical function performs one-hot-encoding
y_train <- to_categorical(y_train,10)
y_test<- to_categorical(y_test,10)




## Multiclass Logistic Regression using Stochastic Gradient Descent




## objective function
objective_func <-  function(beta, X, Y)
{
  # 
  # Definition:     objective_func trains the multiclass logistic regression 
  #     Params:     beta is the weight that paramatrized the linear functions in order to map X to Y.
  #                 X is the features used to train the softmax function
  #                 Y is the classes used to train the softmax function
  #     Return:     returns the cross entropy loss function 
  
  numRows <-  dim(X)[1]
  numFeatures <-  dim(X)[2]
  cost <-  0.0
  
  for (i in 1:numRows) ## softmax acitvation function over rows
  {
    xi <-  X[i,]
    yi <-   Y[i,]
    # print(dim(X))
    # print(dim(beta))
    dotProds <- xi %*% beta
    terms <-  exp(dotProds)
    probs <-  terms / sum(terms)
    k <-  which.max(yi)
    cost <- cost + log(probs[k])
    
  }
  
  return(-cost)
  
  
}



# A <-  as.matrix(rbind(c(3, 6, 7), c(5, -3, 0)))
# B <-  as.matrix(rbind(c(1, 1), c(2, 1), c(3, -3)))
# A%*%B


gradient_desc <- function(beta,X,Y)
{
  # """ 
  #   Definition:     gradient_desc performs stochastic gradicent descent in order to funt the best local minima.
  #       Params:     beta is the weight that paramatrizes the linear functions that map X to Y.
  #                   X is the features used to train the softmax function
  #                   Y is the classes used to train the softmax function
  #       Return:     returns the the corresponding weights
  #   
  #   """
  
  numFeatures <-  length(X)                           ## number of features
  numClass <-  length(Y)                              ## number of classes
  grad <-  matrix(0, nrow = numFeatures, ncol = numClass)   ## number of betas shapes as :  (features, classes)
  
  for (k in 1:numClass) ## softmax acitvation function over columns
  {
    # print(dim(X))
    # print(dim(beta))
    dotProds <-  X %*% beta
    terms <-  exp(dotProds)
    probs <-  terms / sum(terms)
    grad[,k] <-  (probs[k] - Y[k])*X
  }
  
  return(grad)
}



## Multilclass Logistic Regression using Stochastic Gradient Descent  function
multiLogReg <- function(X,Y,lr, epochs)
{
  # """ 
  #   Definition:     multiLogReg performs stochastic gradient descent  
  #       Params:     X is the features used to train the softmax function
  #                   Y is the classes used to train the softmax function
  #                   lr is the learning rate that represents the amount the weights will update on the training data
  #                   epochs is how many times forward and backward propagation will be done on the trainind data
  #       Return:     returns the corresponding beta weights and the cross entropy loss. 
  #   
  #   """        
  
  numSamples <-  dim(X)[1]
  numFeatures <-  dim(X)[2]
  allOnes <-  rep(1,numSamples)               ## creating bias term 
  X <-  cbind(X,allOnes)                      ## adding bias term to the original X
  numFeatures <-  numFeatures+1
  
  numClass <-  dim(Y)[2]
  beta <-  matrix(0, nrow = numFeatures, ncol = numClass)          ## initializing beta weights to zeros with same dim as features and classes
  cost <-  rep(0, epochs)                        ## initialize an array in order to store the cross entropy loss per epoch
  
  
  
  for (ep in 1:epochs)
  {
    cost[ep] <-  objective_func(beta, X, Y)       ## computes each cost per epoch
    
    
    for (i in sample(numSamples))  ## randomly iterates over all rows in order to eliminate biases
    {
      beta <-  beta - lr*gradient_desc(beta, X[i,],Y[i,])  ## updates the beta weights 
      
    }
    
    
    
    cat("Epochs: " , ep , " Cost: " ,cost[ep], "\n")
    
  }
  
  
  
  results <-  list(beta=beta, cost_list= cost)
  return(results)
  
}


multinomial_results <- function(xtrain, ytrain, learning_rate, epochs, xtest, ytest)
{
  # """ 
  #   Definition:     multinomial_results trains the neural network and tests it on the test data (unseen data).
  #       Params:     x_train is the features from the training set used to train the softmax function
  #                   y_train is the classes from the training set used to train the softmax function
  #                   learing_rate is the learning rate that represents the amount the weights will update on the training data
  #                   epochs is how many times forward and backward propagation will be done on the training data
  #                   x_test is the features from the test set.
  #                   y_test is the classes from the test.
  #       Return:     returns the time it took to compute each epoch as well as the percentage of correct classifications. 
  #   
  #   """  
  # 
  
  ptm <- proc.time()
  results_training <- multiLogReg(x_train, y_train, learning_rate, epochs)
  time <- proc.time() - ptm
  cat("Time to compute Stochastic Gradient Descent:" , (time[3]/60) , " minutes " ," for " , epochs , " epochs","\n")
  
  beta <- results_training[[1]]
  
  numSamples <- dim(x_test)[1]
  numFeatures <-  dim(x_test)[2]
  
  allOnes <-  rep(1,numSamples)               ## creating bias term 
  X <-  cbind(x_test,allOnes)                      ## adding bias term to the original X
  numCorrect <-  0
  
  for (i  in 1:numSamples)                    ## performs softmax function over test data
  {
    xi <-  X[i,]
    yi <-  y_test[i,]
    dotProds <-  xi %*% beta                         ## apply beta weights previously trained in the neural network
    terms <-  exp(dotProds)
    probs <-  terms / sum(terms)
    k <-  which.max(probs)                      ## return the index with the maximum probability
                                                ## this index represents the highest probability for the 
                                                ## class the classfier predicts its correct class
    
    if (yi[k] == 1)
    {
      numCorrect <- numCorrect + 1
    }
    
    
  }
  
  
  r <-  (numCorrect / numSamples)*100
  cat("Accuracy:" , r , "%", "\n")
  
}


multinomial_results(x_train, y_train, 0.001, epochs=5, x_test, y_test)  




