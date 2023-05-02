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
import math

def obtain_mean_variance(vals):
    mean = sum(vals) / len(vals)
    vsum = 0
    for val in vals:
        vsum = vsum + (val - mean)**2
    variance = vsum / len(vals)

    std = math.sqrt(variance)
    std = np.std(vals)
    return mean, std


sns.set()

y_scaler1 = MinMaxScaler()

label_fontsize = 18
tick_length = 6
tick_width = 1.5
tick_labelsize = 16
legend_fontsize = 16

fig = plt.figure(figsize=(5,5))

def multiprocessing(function,elements, ncores):
    pool = Pool(processes = ncores)
    results = pool.map(function,elements)
    pool.terminate()
    pool.join()

    return results

lines = []
for i in range(4):
    with open(f'../data/total_train_data_{i}.txt') as f:
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
    if max_sim ==0:
        nearest_smiles = 'None'
    #if smiles == 'C1=CC2=C(N=C1)N=C1C(=N2)c2cccc3-c4cccc5-c6cccc1c6N(c45)c23':
    #if smiles == 'CC(C)(C)c1ccc2c(c1)c1cc(C(C)(C)C)ccc1n2-c1ccc(-c2nc(-c3ccc(P(=O)(c4ccccc4)c4ccccc4)cc3)nc(-c3ccc(P(=O)(c4ccccc4)c4ccccc4)cc3)n2)cc1':
    #    sns.kdeplot(sim_list)
    #    plt.show()
    mean_sim=np.mean(np.array(sim_list))
    return max_sim, nearest_smiles

#label_ind  = 0
import pandas as pd
import seaborn as sns
fns = [ f'TADF-likeness_train_data_{i}.txt' for i in range(4)]
ratio = 0.8
#ratio = 0.6


if True:    
    lines = []
    for fn in fns:
        with open(fn) as f:
            lines += f.readlines()
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
    #cutoff_sim = sort_sim_list[int(len(sort_sim_list)*(1-ratio))]
    
    cutoff_sim = 0.6

    tmp = likeness_list[:]
    tmp.sort()
    #cutoff_likeness = tmp[int(len(tmp)*(1-ratio))]
    #print(cutoff_likeness)
    mean, std =obtain_mean_variance(likeness_list)
    print(mean, std, mean- std)
    #sys.exit()
    #print(sum(likeness_list)/len(likeness_list))
    cutoff_likeness = 88.60424263    
    print('likeness cutoff :', cutoff_likeness)
    print('sim cutoff :', cutoff_sim)
    upper_likeness_list, lower_likeness_list = [], []
    upper_sim_list, lower_sim_list = [] , []
    count = 0
    for likeness, sim in zip(likeness_list, sort_sim_list):
        if likeness > cutoff_likeness:
        #if sim < cutoff_sim:
        #if sim > 0.5:
            upper_sim_list.append(sim)
            upper_likeness_list.append(likeness)
            if sim < cutoff_sim:
                count +=1
        else:
            lower_sim_list.append(sim)
            lower_likeness_list.append(likeness)
                
    print('cutoff likeness upper  : ',len(upper_sim_list), count)
    print('min : ',min(upper_sim_list))
    with open('suppoting_sim_data_train.txt','w') as f:
        for ind, smiles, sim, likeness, nearest_smiles in zip(ind_list, sort_smiles_list, sort_sim_list, likeness_list, sort_nearest_smiles_list):
            f.write(f'{ind} {smiles} {likeness} {sim} {nearest_smiles}\n')
            #if likeness > cutoff_likeness:
            #    print(ind, smiles, sim, likeness, nearest_smiles)    
                

    colors = ['violet','silver','springgreen','gold','deepskyblue']
    x_start = 60

    #sns.kdeplot(upper_sim_list, shade=True)
    sns.histplot(upper_sim_list, bins=7, color = 'deepskyblue')
    plt.tick_params(length=tick_length, width=tick_width, labelsize=tick_labelsize, labelcolor='k', color='k')
    plt.xlabel(rf'MTS', fontsize=label_fontsize, color='k')
    plt.ylabel('Count', fontsize=label_fontsize, color='k')
    plt.xticks([0.2,0.4,0.6,0.8,1.0])

    plt.show()

