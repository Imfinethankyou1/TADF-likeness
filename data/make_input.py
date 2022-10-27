from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from descriptastorus.descriptors import rdNormalizedDescriptors
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import *
from rdkit.Chem import rdMolDescriptors
from multiprocessing import Pool
import numpy as np
import torch
import pandas as pd
import random
random.seed(0)
np.random.seed(seed=0)
#generator = MakeGenerator(("RDKit2D",))
generator = rdNormalizedDescriptors.RDKit2DNormalized()
#columns = []
#for val in generator.columns:
#    if not 'fr' in val[0]:
#        columns.append(val)
#generator.columns = columns
#print(columns)
#sys.exit()

ind_list = random.sample([i for i in range(200)], 47)

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
    if ',' in smiles[-1]:
        smiles = smiles[:-1]
    mol = Chem.MolFromSmiles(smiles)
    try:
        properties = generator.process(smiles)[1:]

        #print(type(properties))
        #sys.exit()
    except:
        return ''
    a= np.array(properties)
    a[np.isnan(a)] = 0
    properties = list(a)
    return properties

def make_results_fp(line):

    properties = []
    if len(line.strip().split()) == 1:
        smiles = line.strip().split(',')[1]
    else:
        smiles = line.strip().split()[1]
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
        fp = list(fp)
    except:
        return ''
    return fp

def rm_duple_from_train(filename, ref_smiles_list):
    with open(f'../data_220615/{filename}') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(line.split()[1]))
        if not smiles in ref_smiles_list:
            new_lines.append(line)
    with open(filename,'w') as f:
        for line in new_lines:
            f.write(line)
    print(filename, len(lines), len(new_lines))    




def make_data(filename, target,input_type):
    if isinstance(filename,str):
        with open(filename) as f:
            lines = f.readlines()
    else:
        lines = filename
    #if ',' in lines[1]:
        
    test_y = []
    count = 0
    if input_type =='descriptor':
        test_x = multiprocessing(make_results,lines, 16)
    elif input_type =='fp':
        test_x = multiprocessing(make_results_fp,lines, 16)
    else:
        print('input_type error')
    smiles_list = []
    for i in range(len(test_x)):
        test_y.append(target)
        if ',' in lines[i]:
            smiles_list.append(lines[i].split(',')[1])
        else:            
            smiles_list.append(lines[i].split()[1])

    return test_x, test_y, smiles_list

def make_npz_file(fn, output, input_type):
    data = {'id':[],'prop':[], 'feats':[],'smiles':[]}
    with open(fn) as f:
        lines = f.readlines()
    x_list, _, smiles_list = make_data(fn, 0, input_type)
    for i in range(len(lines)):
        if not isinstance(x_list[i], str):
            if ',' in lines[i]:
                idx = lines[i].split(',')[0]
            else:
                idx = lines[i].split()[0]
            #print(idx)
            data['id'].append(idx)
            y = float(0.0)
            data['prop'].append(y)
            data['feats'].append(x_list[i])
            data['smiles'].append(smiles_list[i])
    with open(output,'wb') as f:
        pickle.dump(data,f)

def make_data_oled(filename, output_file):
    with open(filename) as f:
        lines = f.readlines()
    #lines = random.sample(lines, 10000)
    test_x = []
    test_y = []
    count = 0
    new_lines = []
    for line in lines:
        TADF = False
        elements = line.strip().split(',')
        if 2.0 <= float(elements[4]) < 3.0:
            if float(elements[4]) - float(elements[5]) < 0.2:
                TADF = True # Pubchem
                count +=1
        #Pub = False
        if TADF:
            new_line = f'{int(elements[0])+500} {elements[1]} 1\n'
            new_lines.append(new_line)
    with open(output_file, 'w') as f:         
        for line in new_lines:
            f.write(line)

def sum_TADF_data(fname1, fname2):
    with open(fname1) as f:
        lines = f.readlines()
    with open(fname2) as f:
        lines += f.readlines()
    smiles_list = []
    for line in lines:
        smiles = line.split()[1]
        smiles_list.append(smiles)
    smiles_list = list(set(smiles_list))
    i = 0
    with open('total_TADF.txt','w') as f:
        for smiles in smiles_list:
            f.write(f'{i} {smiles} 1\n')
            i+=1

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
        x_list, _, smiles_list = make_data(new_lines, 0, input_type='descriptor')
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
    if True:
        fn = 'vis_chromophore_pretrain.txt'
        make_k_fold(fn, 5, 'vis_chromophore_pretrain')
        #fn = 'total_train_data.txt'
        #make_k_fold(fn, 5, 'total_train_data')
