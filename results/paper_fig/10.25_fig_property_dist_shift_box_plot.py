import glob
import rdkit
from rdkit import Chem
import random
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import math 
from matplotlib.ticker import MaxNLocator

random.seed(0)
fns = ['./total_QC_data.txt']



homo_c = -6.5
lumo_c = -1
s1_lc = 1.6
s1_uc = 3.1
est_c = 0.4

smiles_list = []

total_num = 10
interval = 10

homo_list = [[] for i in range(total_num)]
lumo_list = [[] for i in range(total_num)]
s1_list = [[] for i in range(total_num)]
est_list = [[] for i in range(total_num)]

def obtain_mean_variance(vals):
    mean = sum(vals) / len(vals)
    vsum = 0
    for val in vals:
        vsum = vsum + (val - mean)**2
    variance = vsum / len(vals)

    std = math.sqrt(variance)
    std = np.std(vals)
    return [mean, std]

def read_data(filename):
    with open(filename) as f:
        pub_lines = f.readlines()
    pub_homo, pub_lumo, pub_s1, pub_st = [[] for i in range(4)]
    for line in pub_lines:
        homo, lumo, s1, t1, log_f = [ float(val) for val in line.strip().split()[2:7] ]
        pub_homo += [homo]
        pub_lumo += [lumo]
        pub_s1 += [s1]
        pub_st += [s1-t1]
    total = len(pub_lines)
    
    pub_ps =  [ps for ps in [pub_homo,pub_lumo, pub_s1, pub_st] ]
    return pub_ps

tadf_ps = read_data('./total_train_QC_data.txt')

for fn in fns:
    with open(fn) as f:
        lines = f.readlines()
    for line in lines:
        homo, lumo, s1, t1, f, score = [float(val) for val in line.split()[2:]]
        est = s1-t1
        i = int(score/interval)
        if  i == 9 and est > 0.4:
            print(line)


        homo_list[i].append(homo)
        lumo_list[i].append(lumo)
        s1_list[i].append(s1)
        est_list[i].append(est)

#sys.exit()
colors = ['y','orange','r','purple','black']


homo_list =  [homo_list[i]  for i in range(total_num)]
lumo_list =  [lumo_list[i]  for i in range(total_num)]
s1_list =  [s1_list[i]  for i in range(total_num)]
est_list = [est_list[i]  for i in range(total_num)]


properties = [homo_list, lumo_list, s1_list, est_list]

import matplotlib
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as ticker

label_fontsize = 20
tick_length = 6
tick_width = 1.5
tick_labelsize = 16
legend_fontsize = 16



values =  [f'{50+int(i*interval)}-{50+int((i+1)*interval)}' for i in range(5)] +['TADF']
labels = ['HOMO (eV)', 'LUMO (eV)', '$E(S_{1})$', '$\Delta E_{ST}$']
colors = ['y','orange','r','purple','black']
ylim_list = [[-7.7,-1.9], [-3.0,1.65], [0.5, 6.5], [-.06, 2.2] ]

fig = plt.figure(figsize=(15,12))

for i in range(4):
    ax =plt.subplot(2,2,i+1)
    sample_props = properties[i][5:]+[tadf_ps[i][5:]]

    box = plt.boxplot(sample_props, notch=False, patch_artist=True, labels=values)
    colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink','ivory']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.xlabel('TADF-likeness', fontsize=label_fontsize)
    plt.ylabel(f'{labels[i]}', fontsize=label_fontsize, color='k')
    plt.tick_params(length=tick_length, width=tick_width, labelsize=tick_labelsize, labelcolor='k', color='k')
    plt.ylim(ylim_list[i])
    ax.yaxis.set_major_locator(ticker.LinearLocator(5)) 
    plt.tight_layout()
plt.show()
