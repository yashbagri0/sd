baking_time=0.5  # Start with a guess for baking time (the weight)
oven_temp=0.5    # This is fixed (input) (this can be a variable as well, but let's imagine that you got a vision and now know the correct time)
perfect_cookie_score=0.8  # Our goal is to get this perfect score
adjustment=0.001  # Tiny changes we make to baking time each time

# Iteratively adjust baking time
for iteration in range(1101):
    # Bake a batch of cookies
    cookie_quality=oven_temp * baking_time  # Prediction: how good are the cookies? (input * weight, as be discussed above, i.e., front propagation)
    error=(cookie_quality - perfect_cookie_score) ** 2  # How far off are we? (MSE)

    # Print current status
    print(f"Error: {error} | Cookie Quality: {cookie_quality}")

    # Test increasing baking time
    try_up_quality=oven_temp * (baking_time + adjustment)  # Try baking a bit longer
    up_error=(perfect_cookie_score - try_up_quality) ** 2  # Error for baking longer

    # Test decreasing baking time
    try_down_quality=oven_temp * (baking_time - adjustment)  # Try baking a bit shorter
    down_error=(perfect_cookie_score - try_down_quality) ** 2  # Error for baking shorter

    # Decide whether to bake longer or shorter based on smaller error
    if down_error < up_error:  # If shorter baking time gives better cookies
        baking_time=baking_time - adjustment
    elif down_error > up_error:  # If longer baking time gives better cookies
        baking_time=baking_time + adjustment