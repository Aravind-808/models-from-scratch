import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import matplotlib.pyplot as plt
from utils import itoc, ctoi, init_tensor_array, init_charint_map

names = open(r"character-level-lm\names.txt").read().splitlines()

N, char_array = init_tensor_array(names)
char_to_int, int_to_char = init_charint_map(char_array)

for name in names:
    tokens = ["<>"] + list(name) + ["<>"]
    for c1, c2 in zip(tokens, tokens[1:]):
        c1_int = char_to_int[c1]
        c2_int = char_to_int[c2]
        N[c1_int,c2_int] +=1

# print(N)
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

pre = (N+1).float()
pre/=pre.sum(1, keepdims=True)

gen = torch.Generator().manual_seed(628)
for i in range(10):
    out = []
    idx = 0
    while True:
        p = pre[idx]

        idx = torch.multinomial(p, num_samples=1, generator=gen, replacement=True).item()
        out.append(int_to_char[idx])
        if(idx==0):
            break
    print(''.join(out))

log_likelihood = 0.0
n = 0

for name in names[:5]:
    tokens = ["<>"] + list(name) + ["<>"]
    for c1, c2 in zip(tokens, tokens[1:]):
        c1_int = char_to_int[c1]
        c2_int = char_to_int[c2]
        prob = pre[c1_int, c2_int]

        log_prob = torch.log(prob)
        log_likelihood+=log_prob
        n+=1
        print(f"{c1}{c2} {prob:.3f} {log_prob:.3f}")

print(f"log likelihood = {log_likelihood}")

nll = -log_likelihood/n   
print(f"log likelihood (avg)= {nll}") # the lower the number is, the better the model is.


'''
Observations: With respect to the names.txt dataset, comparing the performance of the bigram and trigram models, i can observe that the 
trigram model performs way worse. My bet is that its because of data sparsity. with just 32k names, there arent that many trigrams that occur naturally.

Let me give you an example: the max value of N (which calculates the frequency of bigram and trigrams) in bigram.py was around 6600. This means a certain bigram
occoured in the dataset 6000 times. Compare this to the max N value of trigram.py, which is just around 1700.

Log likelihood (avg) for bigram is ~2.4 and trigram is ~2.1. Although the trigram model seems to be better on paper, idk. the names generated 
by the bigram model just sounds more namelike....
'''