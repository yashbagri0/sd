import numpy as np  # pip install numpy

weights = np.array([0.5, 0.48, -0.7])  # For every input, we've a different weight
lr = 0.1
streetlights = np.array(
    [[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1]]
)  # 1 is on, 0 is off

walk_vs_stop = np.array(
    [0, 1, 0, 1, 1, 0]
)  # actual results. corresponds with the streetlights. eg: 0 means stopped, so people stopped at 1,0,1

for iteration in range(40):
    error_for_all_lights = 0
    for row_index in range(len(walk_vs_stop)):  # Loop through all the rows of the streetlights (each row is one data point)
        input = streetlights[row_index]  # Get the input features (streetlight pattern) for this data point. eg, streetlight[0] gives 1,0,1
        goal_prediction = walk_vs_stop[row_index]  # Get the goal prediction (whether people walk or stop) for this data point. walk_vs_stop[0] is 0
        prediction = input.dot(weights)  # matrix multiplication for forward propagation(earlier we were just multiplying) (1,3) x (1,1)=(1,1) sizeof matrix
        error = (goal_prediction - prediction) ** 2  # MSE
        error_for_all_lights += error
        direction_and_amount = (prediction - goal_prediction) * input  # we get our error (1,1) - (1,1)=(1,1) x (1,3)=(1,3) size
        weights -= lr * direction_and_amount  # calc weights 1 x (1,3)=(1,3)

        print(f"Prediction: {prediction}")
        print(f"Error: {error_for_all_lights} \n")
print(weights)
