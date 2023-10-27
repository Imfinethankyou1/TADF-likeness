import os
import random
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from data_sim_messure import run_sim_messure
from EF_analysis import EF_cal
from EF_final import run_EF_final
from make_database import make_npz_file
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdDepictor
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

random.seed(0)

smiles_list = []
fp_list = []

with open("total_TADF_data.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        word = line.strip().split()
        # _, smiles, _ = word
        smiles = word[1]
        smiles_list.append(smiles)

smiles2fp = {}
smiles2line = {}
assert len(smiles_list) == len(lines)
for smiles, line in zip(smiles_list, lines):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    fp_list.append(arr)
    smiles2fp[smiles] = fp
    smiles2line[smiles] = line

assert len(smiles_list) == len(list(smiles2line.keys()))

pair_smiles2sim = {}
low_sim_smiles = ""
smiles2max_sim = {}
for smiles in smiles_list:
    pair_smiles2sim[smiles] = {}
    for other_smiles in smiles_list:
        if smiles != other_smiles:
            sim = DataStructs.TanimotoSimilarity(
                smiles2fp[smiles], smiles2fp[other_smiles]
            )
            pair_smiles2sim[smiles][other_smiles] = sim
    sim_list = list(pair_smiles2sim[smiles].values())
    max_sim = max(sim_list)
    smiles2max_sim[smiles] = max_sim

smiles_list.sort(key=lambda x: smiles2max_sim[x], reverse=True)


print("max_sim : ", smiles2max_sim[smiles_list[0]])

random.shuffle(smiles_list)


def multiprocessing(function, elements, ncores):
    results = []
    with Pool(ncores) as p:
        n = len(elements)
        with tqdm(total=n) as pbar:
            for result in p.map(function, elements):
                results.append(result)
                pbar.update()

    return results


def cal_max_sim(smiles):
    finger = AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(smiles), 2, nBits=1024
    )
    sim_list = [DataStructs.TanimotoSimilarity(finger, x) for x in origin_finger]
    max_sim = max(sim_list)
    return max_sim


fp_list = []
max_sim = 100
len_test_list = 0
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    fp_list.append(arr)

print("total data num : ", len(fp_list))
target_sim = 0.7
total_smiles_list = smiles_list

while max_sim >= target_sim:
    k = 6
    model = KMeans(n_clusters=k)
    model.fit(fp_list)
    label = model.labels_

    model = KMeans()
    data_list = [[] for i in range(k)]

    for l, smiles in zip(label, total_smiles_list):
        data_list[l].append(smiles)

    count = 0
    total = len(lines)
    train_list = [[] for i in range(k)]

    for i in range(k):
        test_list = []
        train_list[i] = []
        if i == k - 1:
            for j in range(len(data_list[i])):
                test_list.append(data_list[i][j])
        else:
            for j in range(len(data_list[i])):
                train_list[i].append(data_list[i][j])

    total_train_list = []
    idx = 0
    for i in range(k - 1):
        total_train_list += train_list[i]
        with open(f"train_clustering_{i}.txt", "w") as f:
            for line in train_list[i]:
                f.write(f"{idx} {line}\n")
                idx += 1

    origin_finger = [
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=1024)
        for smiles in total_train_list
    ]

    smiles_list = []
    for smiles in test_list:
        smiles_list.append(smiles)

    max_sim_list = list(multiprocessing(cal_max_sim, smiles_list, 8))

    len_test_list = len(test_list)
    print(max(max_sim_list), len_test_list)
    if max(max_sim_list) < target_sim and len_test_list >= 20:
        for i in range(k - 1):
            fn = f"train_clustering_{i}.txt"
            make_npz_file(fn, f'origin/{fn.split(".")[0]}.npz')

        with open("test_clustering.txt", "w") as f:
            for idx, line in enumerate(test_list):
                f.write(f"{idx} {line}\n")
        fn = "test_clustering.txt"
        make_npz_file(fn, f'origin/{fn.split(".")[0]}.npz')

