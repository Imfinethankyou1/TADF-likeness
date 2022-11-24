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

def multiprocessing(function,elements, ncores):
    pool = Pool(processes = ncores)
    results = pool.map(function,elements)
    pool.terminate()
    pool.join()

    return results

fns = ['TADF-likeness-unseen-TADF.txt']

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
    max_sim = max(sim_list)
    if max_sim ==1:
        print(smiles)
    return max_sim

labels = ['other-TADF']
label_ind  = 0
import pandas as pd
import seaborn as sns

for fn in fns:
    
    with open(fn) as f:
        lines = f.readlines()
    if 'pub' in fn:
        lines = lines

    smiles_candidates = []
    for line in lines:
        smiles = line.strip().split()[1]
        smiles_candidates.append(smiles)
        #cal_max_sim(smiles)
    print('start')
    max_sim_list = list(multiprocessing(cal_max_sim, smiles_candidates, 8))

    print('max_sim cal end')
    for_sort = list(zip(max_sim_list,lines))
    for_sort.sort(key=lambda x: x[0])

    sim_list = []
    likeness_list = []

    print('SMILES likeness Sim')
    for sim, line in for_sort:
        sim_list.append(float(sim))
        likeness = float(line.strip().split()[-1])
        smiles = line.strip().split()[0]
        if len(smiles) < 5:
            smiles = line.strip().split()[1] 
        likeness_list.append(likeness)
        

    import matplotlib
    from matplotlib.ticker import MultipleLocator
    label_fontsize = 20
    tick_length = 6
    tick_width = 1.5
    tick_labelsize = 16
    legend_fontsize = 16

    plt.ylim([0.2, 1.1])
    plt.xlim([60, 100.1])
    #plt.yticks([0,0.1,0.2,0.3])
    plt.yticks([0.2,0.4,0.6,0.8,1.0])
    plt.xticks([60,70,80,90,100])
    plt.scatter(likeness_list, sim_list, label=labels[label_ind], marker='^',color='k' )
    #sns.kdeplot(sim_list, label=labels[label_ind])
    label_ind +=1
    plt.xlabel(rf'TADF-likeness', fontsize=label_fontsize)
    plt.ylabel('Tanimoto similarity', fontsize=label_fontsize, color='k')
    #ax2.set_ylabel('$\Delta N_{nc}$ (%)', fontsize=label_fontsize, color='b')
    plt.tick_params(length=tick_length, width=tick_width, labelsize=tick_labelsize, labelcolor='k', color='k')
    matplotlib.rcParams['ytick.major.pad'] = 3
    plt.tight_layout()
    plt.rc('font', size=40)
    #plt.legend(prop={'size': 16}, loc='upper left', ncol=1)
    plt.show()

