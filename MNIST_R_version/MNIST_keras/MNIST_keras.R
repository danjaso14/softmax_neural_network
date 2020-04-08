
suppressMessages(library(reticulate))
suppressMessages(library(tensorflow))
suppressMessages(library(keras))
suppressMessages(library(tidyverse))



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


# The to_categorical function performs one-hot-encoding.
y_train <- to_categorical(y_train,10)
y_test<- to_categorical(y_test,10)


# Specificy neural network architecture,
nn <-  keras_model_sequential() %>%   
  layer_dense(units = 32, activation = "relu", input_shape = ncol(28*28)) %>%
  layer_dense(units = 10, activation = "softmax")

sgd <- optimizer_sgd(lr = 0.01)
compile(nn, loss = "categorical_crossentropy", optimizer = sgd, metrics = "accuracy")

# Train  neural network
fit(nn,  x_train, y_train, epochs = 5, batch_size = 64)


# Run neural network on test set
test_acc <- evaluate(nn, x_test, y_test)
cat("Accuracy: " , round(test_acc$accuracy*100, 2) , "%")





