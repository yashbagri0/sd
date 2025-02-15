import sys, numpy as np
from keras.datasets import mnist #pip install keras

# Load the MNIST dataset (handwritten digits)
(x_train, y_train), (x_test, y_test) = mnist.load_data() #we break the dataset in 2 sets, train(the one we train), and test(we test how well our model has learnt)

# Preprocess the training images: take the first 1000 images, flatten them (28x28 -&gt; 784) and normalize pixel values to range [0, 1] by dividing by 255
images, labels = (x_train[0:1000].reshape(1000, 28*28) / 255, y_train[0:1000])

# Initialize a matrix of zeros for one-hot encoding the labels
one_hot_labels = np.zeros((len(labels), 10)) # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Convert labels (0-9) into one-hot encoded format
for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1 
labels = one_hot_labels  # Replace labels with one-hot encoded labels, so label 3 -&gt; [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

# Preprocess test images: flatten them and normalize pixel values
test_images = x_test.reshape(len(x_test), 28*28) / 255 #normalize the pixels to be between 0-1. 

# One-hot encode test labels
test_labels = np.zeros((len(y_test), 10))
for i, l in enumerate(y_test):
    test_labels[i][l] = 1 #same for test

# Set a random seed for reproducibility
np.random.seed(1) #you can skip if you want

# Define the ReLU (Rectified Linear Unit) activation function
relu = lambda x: (x == 0) * x  # Returns x if x = 0, else returns 0 (fancy way to write a function)

# Derivative of ReLU: 1 if x = 0, else 0
relu2deriv = lambda x: x == 0

# Set hyperparameters
lr = 0.005          # Learning rate
iterations = 350       # Number of training iterations
hidden_size = 40       # Number of nodes in the hidden layer (layer 1)
pixels_per_image = 784 # Number of input nodes (28x28 pixels)
num_labels = 10        # Number of output nodes (one per digit 0-9)

# Initialize weights for input-to-hidden and hidden-to-output layers
weights_0_1 = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1 #random weights layer 0
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1 #random weights for layer 1

# Begin training loop, we'll follow SDG
for j in range(iterations):
    error, correct_cnt = (0.0, 0)  # Initialize total error and correct predictions

    # Loop through each training example
    for i in range(len(images)):
        # Forward pass
        layer_0 = images[i:i+1]                 # Input layer (single image)
        layer_1 = relu(np.dot(layer_0, weights_0_1))  # Hidden layer (apply ReLU)
        layer_2 = np.dot(layer_1, weights_1_2)        # Output layer

        # Calculate total error (squared difference between prediction and true label)
        error += np.sum((labels[i:i+1] - layer_2) ** 2) #MSE

        # Check if the network's prediction matches the true label
        correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))

        # Backpropagation: Calculate direction_and_amount (errors at each layer)
        layer_2_direction_and_amount = (labels[i:i+1] - layer_2)               # Error at output layer
        layer_1_direction_and_amount = layer_2_direction_and_amount.dot(weights_1_2.T) * relu2deriv(layer_1)  # Error at hidden layer

        # Update weights using gradients and learning rate
        weights_1_2 += lr* layer_1.T.dot(layer_2_direction_and_amount)  # Update hidden-to-output weights (we transpose so we can multiply the matrix)
        weights_0_1 += lr * layer_0.T.dot(layer_1_direction_and_amount)  # Update input-to-hidden weights

    # Print training progress (iteration number, average error, and accuracy)
    sys.stdout.write("\r" +
        " I:" + str(j) +
        " Error:" + str(error / float(len(images)))[0:5] +
        " Correct:" + str(correct_cnt / float(len(images))))