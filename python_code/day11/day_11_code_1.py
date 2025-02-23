import numpy as np
                
onehots = {}
onehots['cat'] = np.array([1,0,0,0]) #one hot encoding(OHE) [cat, the, dog, sat] -&gt; [1, 0, 0, 0] for presence of a cat
onehots['the'] = np.array([0,1,0,0])
onehots['dog'] = np.array([0,0,1,0])
onehots['sat'] = np.array([0,0,0,1])

sentence = ['the', 'cat', 'sat']
x = onehots[sentence[0]] + \
    onehots[sentence[1]] + \
    onehots[sentence[2]]
print(f"Sent Encoding: {x}")