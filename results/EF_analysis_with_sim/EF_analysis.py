import glob
import os
from multiprocessing import Pool

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import *

#os.system("python data_likeness_gather.py")


data_names = [
    "combine_dataset_results.txt",
]

for dn in data_names:
    print(dn)
    data = pd.read_csv(dn)

    lines = []
    for i in range(5):
        with open(f"../../data/train_clustering_{i}.txt") as f:
            lines += f.readlines()

    smiles_list = []
    for line in lines:
        smiles_list.append(line.strip().split()[1])
    from rdkit import DataStructs

    origin_finger = [
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=1024)
        for smiles in smiles_list
    ]

    def cal_max_sim(smiles):
        finger = AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smiles), 2, nBits=1024
        )
        sim_list = [DataStructs.TanimotoSimilarity(finger, x) for x in origin_finger]
        max_sim = max(sim_list)
        return max_sim
    if False:
        sim_fn ="data_based_on_sim_combine_set.txt"
        sim_data = pd.read_csv(sim_fn)
        smiles2sim = {}
        for smiles, sim in zip(sim_data['smiles'], sim_data['sim']):
            smiles2sim[smiles] = sim

    idx = 0
    TADF_count = 0
    count = 0
    first = True
    for smiles, data_type, sim in zip(data["smiles"], data["type"], data["sim"]):
        s_sim = cal_max_sim(smiles)
        #assert abs(smiles2sim[smiles] - s_sim) < 0.001
        if float(s_sim) < 0.7 and data_type != 'Chro':
            if idx == 0:
                print('1st sim',sim, s_sim)
            if data_type == "TADF":
                TADF_count += 1
            idx += 1
            if idx == 100:
                print("k = 100, TADF_num =", TADF_count, sim)
            if idx == 500:
                print("k = 500, TADF_num =", TADF_count, sim)
            if idx == 1000:
                print("k = 1000, TADF_num =", TADF_count, sim)
            if idx == 10000:
                print("k = 10000, TADF_num =", TADF_count, sim)
                break
        #else:
        #    print(smiles)                

    # print("Total novel: ",TADF_count)
