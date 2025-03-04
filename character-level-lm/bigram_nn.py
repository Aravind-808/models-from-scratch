import torch
import matplotlib.pyplot as plt

names = open(r"character-level-lm\names.txt").read().splitlines()

N = torch.zeros((27,27), dtype=torch.int32)
char_array = sorted(list(set(''.join(names))))

char_to_int = {char: int+1 for (int, char) in enumerate(char_array)}
char_to_int['<>'] = 0 
int_to_char = {index: char for (char, index) in char_to_int.items()}

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

print(xs)
print(ys)
