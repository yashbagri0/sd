import numpy as np, sys
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() # Load training and testing data (images and labels) from MNIST

images, labels = (x_train[0:1000].reshape(1000, 28 * 28) / 255, y_train[0:1000]) <br># Normalize the first 1000 training images to the range [0, 1] and flatten them to 1D (28*28 = 784 pixels per image)

one_hot_labels = np.zeros((len(labels), 10)) # Initialize a one-hot encoded label array with 10 classes (digits 0-9). [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1 # Convert labels into a one-hot encoding, where the correct digit index is set to 1. 3-&gt; [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

labels = one_hot_labels # Update the labels to be one-hot encoded

test_images = x_test.reshape(len(x_test), 28 * 28) / 255 # Normalize the test images and flatten them to 1D (28*28 = 784 pixels per image)

test_labels = np.zeros((len(y_test), 10)) # Initialize a one-hot encoded test label array

for i, l in enumerate(y_test):
    test_labels[i][l] = 1 # Convert the test labels into a one-hot encoding

def tanh(x):
    return np.tanh(x) # Define the tanh activation function, which is used for introducing non-linearity

def tanh2deriv(output):
    return 1 - (output ** 2) # Define the derivative of the tanh function (needed for backpropagation)

def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True) # Define the softmax function for the output layer to convert logits(output values) into probabilities

lr, iterations = (2, 300) # Set the learning rate (lr) and the number of iterations (epochs) for training

pixels_per_image, num_labels = (784, 10) # 784 pixels per image (28x28) and 10 possible labels (digits 0-9)

batch_size = 128 # Batch size for training (how many samples per forward and backward pass). using mini batch GD.

input_rows = 28
input_cols = 28
# Image size: 28x28 pixels

kernel_rows = 3
kernel_cols = 3
num_kernels = 16
# Define the kernel (filter) size: 3x3, and number of kernels: 16 (for the convolutional layer)

hidden_size = ((input_rows - kernel_rows) * (input_cols - kernel_cols)) * num_kernels # Calculate the size of the hidden layer after applying the kernels

kernels = 0.02 * np.random.random((kernel_rows * kernel_cols, num_kernels)) - 0.01 # Initialize the kernels with small random values.

weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1 # Initialize the weights between the hidden layer and output layer

def get_image_section(layer, row_from, row_to, col_from, col_to):
    section = layer[:, row_from:row_to, col_from:col_to]
    return section.reshape(-1, 1, row_to - row_from, col_to - col_from) <br># Function to extract a section of the image given a row and column range, reshapes it for processing (for our kernel)

for j in range(iterations):
    correct_cnt = 0
    # Track the number of correct predictions during each iteration

    for i in range(int(len(images) / batch_size)):
        # Loop through batches of images. essentially divides our images into batches of 128

        batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))
        layer_0 = images[batch_start:batch_end]
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
        # Select a batch of images and reshape them to 28x28

        sects = list() # List to store image sections (submatrices) for convolution

        for row_start in range(layer_0.shape[1] - kernel_rows):
            for col_start in range(layer_0.shape[2] - kernel_cols):
                sect = get_image_section(layer_0, row_start, row_start + kernel_rows, col_start,
                                         col_start + kernel_cols)
                sects.append(sect)
                #extracts the portion of image for element wise multiplication in our kernel
        # Loop over each position in the image and extract a section for convolution

        expanded_input = np.concatenate(sects, axis=1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)
        # Flatten the extracted image sections for our NN

        kernel_output = flattened_input.dot(kernels) #gets the output
        layer_1 = tanh(kernel_output.reshape(es[0], -1)) #apply activation func

        layer_2 = softmax(np.dot(layer_1, weights_1_2))
        # Pass the output of layer_1 through the softmax function to get final probabilities for each class

        for k in range(batch_size):
            labelset = labels[batch_start + k: batch_start + k + 1]
            _inc = int(np.argmax(layer_2[k:k + 1]) == np.argmax(labelset))
            correct_cnt += _inc
        # Compare predicted label with the true label and count correct predictions

        layer_2_direction_and_amount = (labels[batch_start:batch_end] - layer_2) / (batch_size * layer_2.shape[0])
        # Calculate the error (direction_and_amount) in the output layer (used in backpropagation)

        layer_1_direction_and_amount = layer_2_direction_and_amount.dot(weights_1_2.T) * tanh2deriv(layer_1)
        # Backpropagate the error from the output layer to the hidden layer, apply tanh derivative

        weights_1_2 += lr * layer_1.T.dot(layer_2_direction_and_amount)
        # Update weights between hidden and output layers using the gradient

        l1d_reshape = layer_1_direction_and_amount.reshape(kernel_output.shape)
        k_update = flattened_input.T.dot(l1d_reshape)
        kernels -= lr * k_update
        # Update the kernels (filters) using the gradient calculated during backpropagation

    test_correct_cnt = 0
    # Track the number of correct predictions on the test set

    for i in range(len(test_images)):
        layer_0 = test_images[i:i + 1]
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
        # Reshape a test image for processing

        sects = list()
        for row_start in range(layer_0.shape[1] - kernel_rows):
            for col_start in range(layer_0.shape[2] - kernel_cols):
                sect = get_image_section(layer_0, row_start, row_start + kernel_rows, col_start,
                                         col_start + kernel_cols)
                sects.append(sect)
        # Extract sections of the image for convolution

        expanded_input = np.concatenate(sects, axis=1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)
        kernel_output = flattened_input.dot(kernels)
        layer_1 = tanh(kernel_output.reshape(es[0], -1))
        # Convolve and pass through tanh activation

        layer_2 = np.dot(layer_1, weights_1_2)
        # Compute output probabilities

        test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i + 1]))
        # Compare the predicted label with the true label for test data

    if (j % 1 == 0):
        sys.stdout.write(
            "\n" + "I:" + str(j) + " Test-Acc:" + str(test_correct_cnt / float(len(test_images))) + " Train-Acc:" + str(
                correct_cnt / float(len(images))))
        # Print the test and training accuracy every iteration