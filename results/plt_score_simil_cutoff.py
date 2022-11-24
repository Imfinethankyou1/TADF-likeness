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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Rectangle

y_scaler1 = MinMaxScaler()

label_fontsize = 18
tick_length = 6
tick_width = 1.5
tick_labelsize = 16
legend_fontsize = 16

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(6.8,5.1))
#fig, ax = plt.subplots(figsize=(7,5))
#fig = plt.figure(figsize=(10,6))

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
        
    return max_sim, nearest_smiles

#label_ind  = 0
import pandas as pd
import seaborn as sns
#fns = list(glob.iglob('pubchem_sampling_*txt'))
#fns = ['TADF-likeness-unseen_10.31.txt']

fns = ['TADF-likeness-unseen_no_transfer.txt']



#pipeline = make_pipeline(scaler,model)
#scaler = StandardScaler()
k = 5
model = KMeans(n_clusters=k)
likeness_scaler = MinMaxScaler()
ratio = 0.5

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
    likeness_scaler.fit_transform(np.array(likeness_list).reshape(-1,1))
    scaled_likeness_list = likeness_scaler.transform(np.array(likeness_list).reshape(-1,1))
    X = []
    for sim, likeness in zip(sort_sim_list, scaled_likeness_list):
        X.append(np.array([sim,likeness]))
    X = np.array(X)
    model.fit(X)
    label = model.labels_
    likeness_list_ = np.array(likeness_list)

    #plt.plot([cutoff_likeness, cutoff_likeness], [0.2, 1.05])
    X = np.transpose(X)
    for ind, smiles, sim, likeness, nearest_smiles in zip(ind_list, sort_smiles_list, sort_sim_list, likeness_list, sort_nearest_smiles_list):
        print(ind, smiles, sim, likeness, nearest_smiles)    
    likeness_list.sort()
    cutoff_likeness = likeness_list[int(len(likeness_list)*(1-ratio))]
    colors = ['violet','silver','springgreen','gold','deepskyblue']
    for i in range(k):
        plt.scatter(likeness_list_[label == i] , X[0][label == i] , label = i,  linewidths=0.5, edgecolors='k', c=colors[4])
    plt.xlabel(rf'TADF-likeness (TL)', fontsize=label_fontsize, color='k')
    plt.ylabel('Structural similarity (SS)', fontsize=label_fontsize, color='k')
    plt.tick_params(length=tick_length, width=tick_width, labelsize=tick_labelsize, labelcolor='k', color='k')
    #plt.plot([60, 100], [cutoff_sim, cutoff_sim])
    print('cutoff 1. sim 2.likeness', cutoff_sim, cutoff_likeness)
    ax.add_patch(Rectangle((cutoff_likeness, 0.2), 100.1-cutoff_likeness, 0.9,alpha=0.3, color= 'lime'))
    ax.add_patch(Rectangle((60, cutoff_sim), 50, 1.05-cutoff_sim,alpha=0.3, color= 'lightcoral'))
    plt.text(61, cutoff_sim+0.01, "40 % <",fontsize=13, weight='bold')    
    plt.text(cutoff_likeness+1, 0.22, "40 % <",fontsize=13, weight='bold')    
    matplotlib.rcParams['ytick.major.pad'] = 3
    plt.rc('font', size=40)
    #plt.legend()
    #plt.legend(prop={'size': 16}, loc='upper left', ncol=1)

    plt.ylim([0.2, 1.05])
    plt.xlim([60, 100.1])
    plt.yticks([0.2,0.4,0.6,0.8,1.0])
    plt.xticks([60,70,80,90,100])
    plt.tight_layout()

    plt.show()



    if False:
        fn = 'TADF-likeness-chromophore.txt'
        with open(fn) as f:
            lines = f.readlines()
        smiles_candidates = []
        #mean_TADF_likeness = sum(likeness_list)/len(likeness_list)
        #mean_sim = sum(sort_sim_list)/len(sort_sim_list)
        #print('TADF-likeness(mean)', mean_TADF_likeness)
        #chromophore_likeness_ref = []
        #for line in lines:
        #    smiles = line.strip().split()[1]
        #    smiles_candidates.append(smiles)
        #    likeness = float(line.strip().split()[2])
        #    chromophore_likeness_ref.append(likeness)

        #mean_likeness = sum(chromophore_likeness_ref)/len(chromophore_likeness_ref)
        #median_likeness = np.median(chromophore_likeness_ref)

        pub_sim_list = list(multiprocessing(cal_max_sim, smiles_candidates, 8))
        pub_mean = sum(pub_sim_list)/len(pub_sim_list)
        pub_median = np.median(pub_sim_list)
        
        plt.plot([70, 100],[pub_mean, pub_mean])
        plt.plot([mean_likeness,mean_likeness], [0.2,1.1])
        plt.plot([mean_TADF_likeness,mean_TADF_likeness], [0.2,1.1])
        print('pub_mean : ', pub_mean, len(pub_sim_list), mean_likeness, pub_median, median_likeness)
