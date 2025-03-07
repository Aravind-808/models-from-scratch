import matplotlib.pyplot as plt
import torch

names = open(r"character-level-lm\names.txt",'r').read().splitlines()
charset = sorted(list(set("".join(names))))

# for name in names[:3]:
#     for ch1, ch2, ch3 in zip(name, name[1:], name[2:]):
#        print(ch1,ch2,ch3)


ctoi = {char: index+1 for index, char in enumerate(charset)}
ctoi["<>"] = 0
itoc = {index: char for char, index in ctoi.items()}

N = torch.zeros((27,27,27), dtype=torch.float32)

for name in names:
    padded_name = ["<>"] + list(name) + ["<>"]
    for ch1, ch2, ch3 in zip(padded_name, padded_name[1:], padded_name[2:]):
        n1 = ctoi[ch1]
        n2 = ctoi[ch2]
        n3 = ctoi[ch3]
        N[n1,n2,n3] +=1
# print(N.max())
precompute = (N+1).float()
precompute/=precompute.sum(2, keepdim=True)

# print(precompute)

gen = torch.Generator().manual_seed(3323298)
for i in range(5):
    idx = 0
    out = []
    while(True):
        p = precompute[idx]
        nidx = torch.multinomial(p.flatten(), num_samples=1, replacement=True).item()
        r,c = divmod(nidx, 27)
        out.append(itoc[c])
        if c ==0:
            break
    print("".join(out))
# print(ctoi)
# print(itoc)