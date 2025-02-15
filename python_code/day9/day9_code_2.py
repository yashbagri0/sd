if (j % 10 == 0 or j == iterations - 1):  # Run the test evaluation every 10 iterations or at the final iteration
    error, correct_cnt = (0.0, 0)  # Reset test error and correct count for this evaluation
    for i in range(len(test_images)):  # Loop through all test examples
        layer_0 = test_images[i:i + 1]  # Input layer: a single test image (shape: 1 × 784)
        layer_1 = relu(np.dot(layer_0, weights_0_1))  # Hidden layer: apply ReLU activation
        layer_2 = np.dot(layer_1, weights_1_2)  # Output layer: calculate predictions

        # Compute error (squared difference between true label and predicted output)
        error += np.sum((test_labels[i:i + 1] - layer_2) ** 2)

        # Check if the predicted label matches the true label
        correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i + 1]))

    # Print the test error and accuracy
    sys.stdout.write(" Test-Err:" + str(error / float(len(test_images)))[0:5] + \
                        " Test-Acc:" + str(correct_cnt / float(len(test_images))))
    print()  # Move to the next line for readability