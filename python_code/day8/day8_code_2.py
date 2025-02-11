weight=0.5           # Initial guess for the weight (starting point/baking time)
input=0.5            # Fixed input (oven temperature)
goal_pred=0.8        # Desired prediction (perfect cookies score)

for iteration in range(20):  # Loop for 20 iterations
    pred=input * weight    # Calculate prediction based on the current weight (forward propagation)
    error=(pred - goal_pred) ** 2  # Calculate how wrong the prediction is (MSE)
    direction_and_amount=(pred - goal_pred) * input  # Gradient. It tells us how far off we are and in which direction to move
    weight=weight - direction_and_amount  # Update the weight based on gradient
    print("Error:" + str(error) + " Prediction:" + str(pred))