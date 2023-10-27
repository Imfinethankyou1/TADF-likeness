import glob

# with open('../results/virtual_screening_data_gen/target_TADF_likeness.txt') as f:
#    lines = f.readlines()

# with open('../results/TADF-likeness-unseen-TADF_v1.txt') as f:
#    lines = f.readlines()

lines = []
fns = list(glob.iglob("*txt"))
# fns += ['../results/TADF-likeness-unseen-TADF.txt']
fns.remove("target_TADF_likeness.txt")
for fn in fns:
    with open(fn) as f:
        new_lines = f.readlines()
    lines += new_lines

print("Load data end")

smiles2likeness = {}
for line in lines:
    # print(line)
    label, smiles, likeness = line.strip().split()[:2] + [line.strip().split()[-1]]
    smiles2likeness[smiles] = float(likeness)
print("Make dict end")

new_key_list = list(smiles2likeness.keys())
new_key_list.sort(key=lambda x: smiles2likeness[x], reverse=True)

print("total Num : ", len(new_key_list))
smiles2rank = {}
for idx, key in enumerate(new_key_list):
    smiles2rank[key] = idx

with open("total_gen_smiles.txt", "w") as f:
    # for idx, key in enumerate(list(smiles2rank.keys())):
    for idx, key in enumerate(new_key_list):
        f.write(f"{idx} {key} {smiles2likeness[key]}\n")

if False:
    import pickle

    with open("smiles2likeness.pickle", "wb") as f:
        pickle.dump(smiles2likeness, f)

    with open("smiles2rank.pickle", "wb") as f:
        pickle.dump(smiles2rank, f)


# with open('../results/TADF-likeness-unseen-TADF_v1.txt') as f:
with open("target_TADF_likeness.txt") as f:
    lines = f.readlines()

print("Rearrangement end")
rank_list = []
label_list = []
for line in lines:
    label, smiles, likeness = line.strip().split()[:2] + [line.strip().split()[-1]]
    rank = smiles2rank[smiles]
    rank_list.append(rank)
    #print(smiles2likeness[smiles])
    label_list.append(label)

# label_list = ['F1', 'J1', 'J2', 'L1']
# print(smiles2rank)
for idx, rank in enumerate(rank_list):
    print(f"{label_list[idx]} rank : ", rank + 1)
import sys

sys.exit()
for idx, rank in enumerate(rank_list):
    if rank < 100:
        top100 = idx
    if rank < 1000:
        top1k = idx
    if rank < 10000:
        top10k = idx
    if rank < 100000:
        top100k = idx

print("0.1k, 1k, 10k, 100k")
print(top100, top1k, top10k, top100k)
