import numpy as np

weights = np.array([0.01389228, 1.0138147, -0.01599277])

new_streetlights = np.array(
    [
        [1, 0, 1],  # should be 0
        [0, 0, 1],  # should be 0
        [1, 1, 1],  # should be 1
    ]
)

for new_input in new_streetlights:
    prediction = new_input.dot(weights)  # Calculate the dot product of new input and weights

    # If the prediction is closer to 1, we consider it as "walk", else "stop"
    if prediction >= 0.5:
        print("Prediction: Walk (1)")
    else:
        print("Prediction: Stop (0)")

    print(f"Prediction Value: {prediction}\n")
