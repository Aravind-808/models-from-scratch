'''
IGNORE THIS. THIS IS ONLY FOR PRACTICE. CODE THAT WORKS ARE IN DEDICATED FILES.
'''


import torch
import matplotlib.pyplot as plt

names = open(r"character-level-lm\names.txt").read().splitlines()

# print(names[:10])

# print(max(len(name) for name in names))
# print(min(len(name) for name in names))

# name_longest = [names[i] for i in range(len(names)) if len(names[i]) == 15]
# name_shortest = [names[i] for i in range(len(names)) if len(names[i]) == 2]

# print(name_longest)
# print(name_shortest)

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
# print(a)

# creating character lookup table
N = torch.zeros((27,27), dtype=torch.int32)
char_array = sorted(list(set(''.join(names))))

# print(char_array)

# char_to_int = {char: int for (int, char) in enumerate(char_array)}
# char_to_int['<S>'] = 26
# char_to_int['<E>'] = 27
# print(char_to_int)

# for name in names:
#     tokens = ["<S>"] + list(name) + ["<E>"]
#     for c1, c2 in zip(tokens, tokens[1:]):
#         c1_int = char_to_int[c1]
#         c2_int = char_to_int[c2]
#         N[c1_int,c2_int] +=1
# print(N)

# int_to_char = {char: int for (int, char) in char_to_int.items()}
# print(int_to_char)
# plt.figure(figsize=(16,16))
# plt.imshow(N, cmap='Blues')
# for i in range(27):
#     for j in range(27):
#         s = int_to_char[i] + int_to_char[j]
#         plt.text(i,j, s, ha = "center", va = "bottom")
#         plt.text(i, j, N[i,j].item(), ha = "center", va = "top")
# plt.axis('off')
# plt.show()

char_to_int = {char: int+1 for (int, char) in enumerate(char_array)}
char_to_int['<>'] = 0   
# print(char_to_int)

for name in names:
    tokens = ["<>"] + list(name) + ["<>"]
    for c1, c2 in zip(tokens, tokens[1:]):
        c1_int = char_to_int[c1]
        c2_int = char_to_int[c2]
        N[c1_int,c2_int] +=1
# print(N)

int_to_char = {index: char for (char, index) in char_to_int.items()}


# print(N[0])
# probabilities = N[0].float()
# probabilities = probabilities/probabilities.sum()
# gen = torch.Generator().manual_seed(23)
# multinomial = torch.multinomial(probabilities, num_samples=10, generator=gen, replacement=True)

gen2 = torch.Generator().manual_seed(3782733628)
for i in range(5):
    out = []
    idx = 0
    while True:
        p = N[idx].float()
        p = p/p.sum()

        mul = torch.multinomial(p, num_samples=1, generator=gen2, replacement=True).item()
        out.append(int_to_char[mul])
        if(mul==0):
            break
    print(''.join(out))

# print(multinomial)
# for idx in multinomial:
#     print(int_to_char[idx.item()], end = " ")
# p = torch.rand(3, generator=gen)
# p = p/p.sum()
# print(p)
