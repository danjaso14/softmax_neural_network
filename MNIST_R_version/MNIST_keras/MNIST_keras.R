
suppressMessages(library(reticulate))
suppressMessages(library(tensorflow))
suppressMessages(library(keras))
suppressMessages(library(tidyverse))



mnist <- dataset_mnist()
x_train_m <- mnist$train$x
y_train_m <- mnist$train$y
x_test_m <- mnist$test$x
y_test_m <- mnist$test$y


# reshape
x_train_m <- array_reshape(x_train_m, c(nrow(x_train), 784))
x_test_m <- array_reshape(x_test_m, c(nrow(x_test), 784))

# rescale
x_train_m <- x_train_m / 255
x_test_m <- x_test_m / 255


# The to_categorical function performs one-hot-encoding.
y_train_m <- to_categorical(y_train_m,10)
y_test_m <- to_categorical(y_test_m,10)


# Specificy neural network architecture,
nn <-  keras_model_sequential() %>%   
  layer_dense(units = 32, activation = "relu", input_shape = ncol(28*28)) %>%
  layer_dense(units = 10, activation = "softmax")

sgd <- optimizer_sgd(lr = 0.01)
compile(nn, loss = "categorical_crossentropy", optimizer = sgd, metrics = "accuracy")

# Train  neural network
fit(nn,  x_train_m, y_train_m, epochs = 5, batch_size = 64)


# Run neural network on test set
test_acc <- evaluate(nn, x_test_m, y_test_m)
cat("Accuracy: " , round(test_acc$accuracy*100, 2) , "%")





