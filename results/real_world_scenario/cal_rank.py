import glob
import pickle


#smiles2likeness = {}
#smiles2rank = {}

with open('smiles2likeness.pickle', 'rb') as f:
    smiles2likeness = pickle.load(f) 

with open('smiles2rank.pickle', 'rb') as f:
    smiles2rank = pickle.load(f) 

new_key_list = list(smiles2likeness.keys())
new_key_list.sort(key=lambda x : smiles2likeness[x],reverse=True)

print('total Num : ', len(new_key_list))

with open('target_TADF_likeness.txt') as f:
    lines = f.readlines()

print('Rearrangement end')
rank_list = []
for line in lines:
    label, smiles, likeness = line.strip().split()[:2]+[line.strip().split()[-1]]
    rank = smiles2rank[smiles]
    rank_list.append(rank)

label_list = ['F1', 'J1', 'J2', 'L1']

#print(smiles2rank)
for idx, rank in enumerate(rank_list):
    print(f'{label_list[idx]} rank : ',rank+1)
import sys

print('#######TADF-testset start############')

with open('../TADF-likeness-unseen-TADF.txt') as f:
    lines = f.readlines()

rank_list = []
for line in lines:
    label, smiles, likeness = line.strip().split()[:2]+[line.strip().split()[-1]]
    rank = smiles2rank[smiles]
    rank_list.append(rank)

for idx, rank in enumerate(rank_list):
    if rank < 100:
        top100 = idx
    if rank < 1000:
        top1k = idx
    if rank < 10000:
        top10k = idx
    if rank < 100000:
        top100k = idx

print('0.1k, 1k, 10k, 100k')
print(top100, top1k, top10k, top100k)
