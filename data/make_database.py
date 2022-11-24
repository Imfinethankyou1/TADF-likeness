from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from descriptastorus.descriptors import rdNormalizedDescriptors
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import *
from multiprocessing import Pool
import numpy as np
import torch
import pandas as pd
import random
import os
random.seed(0)
np.random.seed(seed=0)
generator = rdNormalizedDescriptors.RDKit2DNormalized()

def multiprocessing(function,elements, ncores):
    pool = Pool(processes = ncores)
    results = pool.map(function,elements)
    pool.terminate()
    pool.join()

    return results

def make_results(line):
    
    properties = []
    if len(line.strip().split()) == 1:
        smiles = line.strip().split(',')[1]
    else:
        smiles = line.strip().split()[1]
    if '.' in smiles:
        return ''
    mol = Chem.MolFromSmiles(smiles)
    try:
        properties = generator.process(smiles)[1:]
    except:
        return ''
    a= np.array(properties)
    a[np.isnan(a)] = 0
    properties = list(a)
    return properties


def rm_duple_from_train(filename, ref_smiles_list):
    with open(f'../data_220615/{filename}') as f:
        lines = f.readlines()

    #smiles_list = []
    #smiles2line = {}
    new_lines = []
    for line in lines:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(line.split()[1]))
        if not smiles in ref_smiles_list:
            new_lines.append(line)
    with open(filename,'w') as f:
        for line in new_lines:
            f.write(line)
    print(filename, len(lines), len(new_lines))    


def make_data(filename, target):
    with open(filename) as f:
        lines = f.readlines()
    test_y = []
    count = 0

    test_x = multiprocessing(make_results,lines, 8)

    smiles_list = []
    for i in range(len(test_x)):
        test_y.append(target)
        smiles_list.append(lines[i].split()[1])

    return test_x, test_y, smiles_list

def make_npz_file(fn, output):
    data = {'id':[],'prop':[], 'feats':[],'smiles':[]}
    with open(fn) as f:
        lines = f.readlines()
    x_list, _, smiles_list = make_data(fn, 0)
    for i in range(len(lines)):
        if not isinstance(x_list[i], str):
            idx=lines[i].split()[0]
            print(idx)
            data['id'].append(idx)
            y = float(0.0)
            data['prop'].append(y)
            data['feats'].append(x_list[i])
            data['smiles'].append(smiles_list[i])
    with open(output,'wb') as f:
        pickle.dump(data,f)

def make_k_fold(fn,k, output):
    with open(fn) as f:
        lines = f.readlines()
    random.shuffle(lines)        
    for j in range(k):
        data = {'id':[],'prop':[], 'feats':[],'smiles':[]}
        new_lines = lines[j::k]
        with open(f'{output}_{j}.txt','w') as f:
            for line in new_lines:
                f.write(line)
        x_list, _, smiles_list = make_data(new_lines, 0)
        for i in range(len(new_lines)):
            if not isinstance(x_list[i], str):
                idx = new_lines[i].split()[0]
                #print(idx)
                data['id'].append(idx)
                y = float(0.0)
                data['prop'].append(y)
                data['feats'].append(x_list[i])
                data['smiles'].append(smiles_list[i])
        with open(f'origin/{output}_{j}.npz','wb') as f:
            pickle.dump(data,f)


if __name__ == '__main__':
    import glob
    import sys
    #filenames = ['unseen-TADF.txt']
    if not os.path.isdir('origin'):
        os.system('mkdir origin/')
    filenames = ['total_train_data.txt','pubchem_200k.txt', 'unseen-TADF.txt' ,'vis_chromophore.txt']
    for i in range(len(filenames)):
        make_npz_file(filenames[i],f'origin/{filenames[i].split(".")[0]}.npz')
    
    #fn = 'total_train_data.txt'
    #make_k_fold(fn, 5, 'total_train_data')
