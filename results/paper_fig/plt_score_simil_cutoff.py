import glob
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import *
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from multiprocessing import Pool
import pandas as pd
import matplotlib
from matplotlib.ticker import MultipleLocator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Rectangle

sns.set()

y_scaler1 = MinMaxScaler()

label_fontsize = 18
tick_length = 6
tick_width = 1.5
tick_labelsize = 16
legend_fontsize = 16

fig = plt.figure(figsize=(7,5))

def multiprocessing(function,elements, ncores):
    pool = Pool(processes = ncores)
    results = pool.map(function,elements)
    pool.terminate()
    pool.join()

    return results

lines = []
for i in range(4):
    with open(f'../total_train_data_{i}.txt') as f:
        lines += f.readlines()
smiles_list = []
for line in lines:
    smiles_list.append(line.strip().split()[1])
from rdkit import DataStructs
origin_finger = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles),2,nBits=1024) for smiles in smiles_list]

def cal_max_sim(smiles):
    finger = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles),2,nBits=1024)
    sim_list = [ DataStructs.TanimotoSimilarity(finger,x) for x in origin_finger]
    #max_sim = max(sim_list)
    max_sim = 0
    for i in range(len(sim_list)):
        sim = sim_list[i]
        if sim > max_sim:
            max_sim = sim
            nearest_smiles = smiles_list[i]
        
    mean_sim=np.mean(np.array(sim_list))
    return max_sim, nearest_smiles

#label_ind  = 0
import pandas as pd
import seaborn as sns
#fns = list(glob.iglob('pubchem_sampling_*txt'))
fns = ['TADF-likeness-unseen_no_transfer.txt']
#fns = ['TADF-likeness-unseen_old.txt']
#fns = ['TADF-likeness-pubchem_no_transfer.txt']
#pipeline = make_pipeline(scaler,model)
#scaler = StandardScaler()
ratio = 0.8

for fn in fns:
    
    with open(fn) as f:
        lines = f.readlines()
    smiles_candidates = []
    for line in lines:
        ind, smiles = line.strip().split()[0:2]
        smiles_candidates.append(smiles)
    max_sim_list = list(multiprocessing(cal_max_sim, smiles_candidates, 8))

    nearest_smiles_list = []
    new_sim_list = []
    for sim, nearest_smiles in max_sim_list:
        new_sim_list.append(sim)
        nearest_smiles_list.append(nearest_smiles)
        
    max_sim_list = new_sim_list
    for_sort = list(zip(max_sim_list,lines,nearest_smiles_list))
    for_sort.sort(key=lambda x: x[0])
    
    sort_sim_list = []
    likeness_list = []
    sort_smiles_list = []
    ind_list = []
    sort_nearest_smiles_list = []
    print('SMILES likeness Sim')
    for sim, line, nearest_smiles in for_sort:
        sort_sim_list.append(float(sim))
        likeness = float(line.strip().split()[-1])
        ind, smiles = line.strip().split()[:2] 
        likeness_list.append(likeness)
        sort_smiles_list.append(smiles)
        sort_nearest_smiles_list.append(nearest_smiles)
        ind_list.append(ind)
    cutoff_sim = sort_sim_list[int(len(sort_sim_list)*(1-ratio))]
    
    tmp = likeness_list[:]
    tmp.sort()
    cutoff_likeness = tmp[int(len(tmp)*(1-ratio))]

    upper_likeness_list = []
    upper_sim_list = []
    for likeness, sim in zip(likeness_list, sort_sim_list):
        if likeness > cutoff_likeness:
            upper_sim_list.append(sim)
            upper_likeness_list.append(likeness)

    colors = ['violet','silver','springgreen','gold','deepskyblue']
    x_start = 60

    #plt.ylim([0.2, 1.05])
    #plt.xlim([x_start, 100.1])
    #plt.yticks([0.2,0.4,0.6,0.8,1.0])
    #sns.kdeplot(upper_sim_list)
    sns.histplot(upper_sim_list, color='springgreen' ,bins=7, shrink=.97 )
    plt.tick_params(length=tick_length, width=tick_width, labelsize=tick_labelsize, labelcolor='k', color='k')
    plt.xlabel(rf'Structral similarity (SS)', fontsize=label_fontsize, color='k')
    plt.ylabel('Count', fontsize=label_fontsize, color='k')

    plt.show()


