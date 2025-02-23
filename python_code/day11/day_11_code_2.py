import numpy as np
import sys

f = open('reviews.txt') #download at https://github.com/iamtrask/Grokking-Deep-Learning/blob/master/reviews.txt
raw_reviews = f.readlines()
f.close()

f = open('labels.txt') #download at https://github.com/iamtrask/Grokking-Deep-Learning/blob/master/labels.txt
raw_labels = f.readlines()
f.close()

tokens = list(map(lambda x: set(x.split(" ")), raw_reviews)) #makes a list of each line as 
#[{"This", "cinematic", "masterpiece", "changed", "my", "life,", "now", "I", "can't", "stop", "speaking", "in", "movie", "quotes", "and", "my", "family", "is", "concerned!"}, 
# {"This", "film", "was", "so", "bad,", "my", "popcorn", "filed", "for", "emotional", "damages"}...]

vocab = set()
for sent in tokens:
    for word in sent:
        if len(word) > 0:
            vocab.add(word) #add all unqiue words in our vocab set

vocab = list(vocab) #change set to list


word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i #make the OHE, eg, {"great": 0, "movie": 1, "this": 2, "was": 3}


input_dataset = list()
for sent in tokens:
    sent_indices = list()
    for word in sent:
        try:
            sent_indices.append(word2index[word])
        except:
            ""
    input_dataset.append(list(set(sent_indices)))
# if [{"great", "movie"}, {"this", "movie"}] and {"great": 0, "movie": 1, "this": 2} -> [[0, 1], [1, 2]]

target_dataset = list()
for label in raw_labels:
    if label == 'positive\n':
        target_dataset.append(1)
    else:
        target_dataset.append(0)
# postive, negative -> 1, 0


np.random.seed(1)

def sigmoid(x):
    return 1/(1 + np.exp(-x)) #range between 0 and 1

lr, iterations = (0.01, 2)
hidden_size = 100
weights_0_1 = 0.2 * np.random.random((len(vocab), hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, 1)) - 0.1

correct, total = (0,0)
for iter in range(iterations):
    for i in range(len(input_dataset) - 1000):
        x, y = (input_dataset[i], target_dataset[i])
        layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))


        layer_2_direction_and_amount = layer_2 - y 
        layer_1_direction_and_amount = layer_2_direction_and_amount.dot(weights_1_2.T)

        weights_0_1[x] -= layer_1_direction_and_amount * lr
        weights_1_2 -= np.outer(layer_1, layer_2_direction_and_amount) * lr


        if np.abs(layer_2_direction_and_amount) < 0.5:
            correct += 1
        total += 1

        if(i % 10 == 9):
            progress = str(i/float(len(input_dataset)))
            sys.stdout.write('\rIter:'+str(iter)\
            +' Progress:'+progress[2:4]\
            +'.'+progress[4:6]\
            +'% Training Accuracy:'\
            + str(correct/float(total)) + '%')

    print()

correct, total = (0, 0)
for i in range(len(input_dataset) - 1000, len(input_dataset)):
    x = input_dataset[i]
    y = target_dataset[i]
    layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
    layer_2 = sigmoid(np.dot(layer_1, weights_1_2))
    if (np.abs(layer_2 - y) < 0.5):
        correct += 1
    total += 1
print(f"Test Accuracy: {correct / float(total)}")