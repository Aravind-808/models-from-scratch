import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

names = open(r"character-level-lm\names.txt").read().splitlines()

gen = torch.Generator().manual_seed(3273827392)
N = torch.zeros((27,27), dtype=torch.int32)
char_array = sorted(list(set(''.join(names))))

char_to_int = {char: int+1 for (int, char) in enumerate(char_array)}
char_to_int['<>'] = 0 
int_to_char = {index: char for (char, index) in char_to_int.items()}

'''
To feed data and labels into the neural network, we first initialize the training sets (labels and data)

The network will return for the input of one character,the probability distribution of the next char.

'''
xs, ys = [],[]

for name in names[:1]:
    tokens = ["<>"] + list(name) + ["<>"]
    for c1, c2 in zip(tokens, tokens[1:]):
        c1_int = char_to_int[c1]
        c2_int = char_to_int[c2]
        xs.append(c1_int)
        ys.append(c2_int)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

# print(xs)
# print(ys)

'''
One hot encoding returns a vector with every value as 0 and the occourence of tensor as 1.
'''
x_encoded = F.one_hot(xs, num_classes=27).float()
print(x_encoded.shape)
# plt.imshow(x_encoded)
# plt.show()
 
W = torch.randn((27,27), generator=gen)

'''
Logits are log counts. when we take the raw dot product of the W and encoded x matrix, we end up
with a tensor containing negative values. Hence, to have values that we can work with, we 
exponentiate the tensor, so that the values are always positive.
'''

# This is an implementation of softmax
'''
softmax(z):
output layer z -> torch.exp(z)/torch.sum(torch.exp(z)) -> probabilities
'''

logits  = x_encoded @ W 
count_logits  = logits.exp()

# Just like we did with the count method, we take the probability
probabilities = count_logits/count_logits.sum(1, keepdims=True)

# results in 1
print(round(probabilities[0].sum().item()))

# Taking "emma" as an example

test = torch.zeros(5)

neg_log = torch.zeros(5)
for i in range(5):
    x = xs[i].item()
    y = ys[i].item()
    print("-------------------------")
    print(f"Bigram: {int_to_char[x]}{int_to_char[y]} (Indices: {x}{y})")
    print(f"Input To the neural network: {x} (alphabet/tag: {int_to_char[x]})")
    print(f"Probabilities assigned for the next character is {probabilities[i]}")
    print(f"Actual Next character: {y} (alphabet/tag: {int_to_char[y]})")
    prob = probabilities[i,y]
    print(f"Probability assigned by the network: {prob.item()}")
    log_likelihood = torch.log(prob)
    neg_log_likelihood = -log_likelihood
    print(f"Negative log likelohood: {neg_log_likelihood}")
    neg_log[i] = neg_log_likelihood

avg_nll = neg_log.mean()
print(f"Average negative log for test word is {avg_nll}")
