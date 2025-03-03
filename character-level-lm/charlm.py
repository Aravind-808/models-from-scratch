import torch
import matplotlib.pyplot as plt

names = open(r"character-level-lm\names.txt").read().splitlines()
'''
print(names[:10])

print(max(len(name) for name in names))
print(min(len(name) for name in names))

name_longest = [names[i] for i in range(len(names)) if len(names[i]) == 15]
name_shortest = [names[i] for i in range(len(names)) if len(names[i]) == 2]

print(name_longest)
print(name_shortest)
'''

# Bigrams
# bigram_freq = {}
for name in names:
    tokens = ["<Start>"] + list(name) + ["<End>"]
    for c1, c2 in zip(tokens, tokens[1:]):
        bigram = (c1, c2)
        # bigram_freq[bigram] = bigram_freq.get(bigram,0) +1
        # print(c1, c2)
    
# print(sorted(bigram_freq.items(), key= lambda kv: kv[1]))

a = torch.zeros((3,5))
print(a)

# creating character lookup table
N = torch.zeros((28,28), dtype=torch.int32)
char_array = sorted(list(set(''.join(names))))
# print(char_array)

char_to_int = {char: int for (int, char) in enumerate(char_array)}
char_to_int['<S>'] = 26
char_to_int['<E>'] = 27
print(char_to_int)

for name in names:
    tokens = ["<S>"] + list(name) + ["<E>"]
    for c1, c2 in zip(tokens, tokens[1:]):
        c1_int = char_to_int[c1]
        c2_int = char_to_int[c2]
        N[c1_int,c2_int] +=1
print(N)

int_to_char = {char: int for (int, char) in char_to_int.items()}
print(int_to_char)
plt.figure(figsize=(32,32))
plt.imshow(N, cmap='Blues')
for i in range(28):
    for j in range(28):
        s = int_to_char[i] + int_to_char[j]
        plt.text(i,j, s)
        plt.text(i, j, N[i,j].item())
plt.show()