import sys, random, math
from collections import Counter
import numpy as np

# Setting random seeds for reproducibility
np.random.seed(1)
random.seed(1)

# Load movie reviews dataset
f = open('reviews.txt')
raw_reviews = f.readlines()
f.close()

# Tokenize each review (split into words)
tokens = list(map(lambda x: (x.split(" ")), raw_reviews))

# Count word frequencies.
wordcnt = Counter()
for sent in tokens:
    for word in sent:
        wordcnt[word] -= 1  # Subtracting 1 to adjust for frequency counting

# Create vocabulary (list of unique words)
vocab = list(set(map(lambda x: x[0], wordcnt.most_common())))

# Mapping words to indices
word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i

# Initialize dataset and concatenated list
concatenated = list()
input_dataset = list()

# Convert words in sentences to their indices and store
for sent in tokens:
    sent_indices = list()
    for word in sent:
        try:
            sent_indices.append(word2index[word])  # Get index of the word from word2index
            concatenated.append(word2index[word])  # Add to concatenated list
        except:
            pass  # Handle any words that are not in the vocabulary
    input_dataset.append(sent_indices)

# Convert concatenated list to a NumPy array for better performance
concatenated = np.array(concatenated)

# Shuffle the input dataset for randomness during training
random.shuffle(input_dataset)

# Hyperparameters: learning rate, number of iterations, hidden layer size, window size, and negative sampling size
lr, iterations = (0.05, 2)
hidden_size, window, negative = (50, 2, 5)

# Initialize weights for input to hidden and hidden to output layers
weights_0_1 = (np.random.rand(len(vocab), hidden_size) - 0.5) * 0.2
weights_1_2 = np.random.rand(len(vocab), hidden_size) * 0

# Initialize target vector for the negative sampling
layer_2_target = np.zeros(negative + 1)
layer_2_target[0] = 1  # Set the first target element to 1 (the positive sample)


# Function to find the most similar words to a target word using cosine similarity
def similar(target='beautiful'):
    target_index = word2index[target]
    scores = Counter()
    for word, index in word2index.items():
        raw_difference = weights_0_1[index] - (weights_0_1[target_index])
        squared_difference = raw_difference * raw_difference
        scores[word] = -math.sqrt(sum(squared_difference))  # Negative for descending order
    return scores.most_common(10)


# Sigmoid activation function for logistic regression
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Training loop
for rev_i, review in enumerate(input_dataset * iterations):
    for target_i in range(len(review)):
        # Randomly sample words for negative sampling
        target_samples = [review[target_i]] + list(
            concatenated[(np.random.rand(negative) * len(concatenated)).astype('int').tolist()])
        #target_samples = [2, 150, 323, 99, 478], where 2 is the index of the target word, and the restare index for
        # negative(wrong) words

        # Context words around the target word (window size(2) on either side) -&gt; lamb whose __________ was white
        left_context = review[max(0, target_i - window): target_i]
        right_context = review[target_i + 1: min(len(review), target_i + window)]

        # Calculate the mean of the context word vectors
        layer_1 = np.mean(weights_0_1[left_context + right_context], axis=0)

        # Perform the forward pass through the hidden to output layer
        layer_2 = sigmoid(layer_1.dot(weights_1_2[target_samples].T))

        # Calculate the error (delta is direction_and_amount) for backpropagation
        layer_2_delta = layer_2 - layer_2_target
        layer_1_delta = layer_2_delta.dot(weights_1_2[target_samples])

        # Update the weights using gradient descent
        weights_0_1[left_context + right_context] -= layer_1_delta * lr
        weights_1_2[target_samples] -= np.outer(layer_2_delta, layer_1) * lr

    # Print progress and check for word similarity every 250 reviews
    if rev_i % 250 == 0:
        sys.stdout.write('\rProgress: ' + str(rev_i / float(len(input_dataset) * iterations)) + " " + str(similar('terrible')))
    sys.stdout.write('\rProgress: ' + str(rev_i / float(len(input_dataset) * iterations)))

# Print the most similar words to 'terrible and beautiful' after training
print(similar('beautiful'))
print(similar('terrible'))