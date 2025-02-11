import numpy as np

np.random.seed(1)  # Setting a random seed for reproducibility, you can ignore if you want


# Define the ReLU activation function: outputs the value if it's positive, otherwise outputs 0
def relu(x):
    return ((x > 0) * x)  # If x > 0, keep it; otherwise, make it 0. We won't entertain any negativity. (max(0, x))


# Define the derivative of ReLU: 1 if output > 0, else 0
def relu2deriv(output):
    return output > 0  # 1 if > 0, else 0


streetlights = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]])

walk_vs_stop = np.array([[1], [1], [0], [0]])

lr = 0.2

# Size of the hidden layer (number of neurons in the middle layer)
hidden_size = 4

# Initializing weights randomly between -1 and 1
# weights_0_1: connects input layer to hidden layer
weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
# weights_1_2: connects hidden layer to output layer
weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

for iteration in range(60):
    layer_2_error = 0  # Tracks the total error in this iteration

    for i in range(len(streetlights)):
        layer_0 = streetlights[i : i + 1]  # Input layer: one row of data (1x3 matrix)
        layer_1 = relu(np.dot(layer_0, weights_0_1))  # Hidden layer: apply weights and ReLU activation
        layer_2 = np.dot(layer_1, weights_1_2)  # Output layer: apply weights to hidden layer

        # Calculate error for the current example
        layer_2_error += np.sum((layer_2 - walk_vs_stop[i : i + 1]) ** 2)

        # Calculate how much to adjust the output layer weights
        layer_2_direction_and_rate = (layer_2 - walk_vs_stop[i : i + 1])  # Difference between prediction and actual value

        # Calculate how much to adjust the hidden layer weights
        layer_1_direction_and_rate = layer_2_direction_and_rate.dot(weights_1_2.T) * relu2deriv(layer_1)
        # Pass error backward through the weights and use ReLU derivative to decide how much to change

        # Update weights: adjust by the error scaled by learning rate
        weights_1_2 -= lr * layer_1.T.dot(layer_2_direction_and_rate)  # Update weights from hidden to output layer
        weights_0_1 -= lr * layer_0.T.dot(layer_1_direction_and_rate)  # Update weights from input to hidden layer

    # Every 10 iterations, print the total error
    if iteration % 10 == 9:
        print(f"Error: {layer_2_error}")
