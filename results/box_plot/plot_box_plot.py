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


homo_c = -6.5
lumo_c = -1
s1_lc = 1.6
s1_uc = 3.1
est_c = 0.4

smiles_list = []

total_num = 4

homo_list = [[] for i in range(total_num)]
lumo_list = [[] for i in range(total_num)]
s1_list = [[] for i in range(total_num)]
est_list = [[] for i in range(total_num)]
lines_list = [[] for i in range(total_num)  ]

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

tadf_ps = read_data('../total_train_QC_data.txt')


fns = ['box_plot_data.txt']

for fn in fns:
    with open(fn) as f:
        lines = f.readlines()
    for line in lines:
        homo, lumo, s1, t1, f, score = [float(val) for val in line.split()[2:]]
        est = s1-t1
        is_append = False
        if score >=50 and score < 60:
            i = 0
            is_append = True
        if score >=70 and score < 80:
            i = 1
            is_append = True
        if score >=90 and score < 95:
            i = 2
            is_append = True
        if score >=95 and score <= 100:
            i = 3
            is_append = True
        if is_append:
            homo_list[i].append(homo)
            lumo_list[i].append(lumo)
            s1_list[i].append(s1)
            est_list[i].append(est)
            lines_list[i].append(line)

for i in range(4):
    if len(lines_list[i]) >100:
        idx_list = [j for j in range(len(lines_list[i]))]
        idx_list = random.sample(idx_list, 100)
        homo_list[i] = np.array(homo_list[i])[idx_list]
        lumo_list[i] = np.array(lumo_list[i])[idx_list]
        s1_list[i] = np.array(s1_list[i])[idx_list]
        est_list[i] = np.array(est_list[i])[idx_list]
        lines_list[i] =  [lines_list[i][idx] for idx in idx_list]
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


values =  ['Low', 'Medium-low', 'Medium-high', 'High'] + ['TADF']
labels = ['HOMO (eV)', 'LUMO (eV)', '$E(S_{1})$', '$\Delta E_{ST}$']
colors = ['y','orange','r','purple','black']
ylim_list = [[-6.7,-4.1], [-3.0,-0.1], [1.7, 5.3], [-0.02, 1.75] ]

fig = plt.figure(figsize=(15,12))

for i in range(4):
    ax =plt.subplot(2,2,i+1)
    sample_props = properties[i]+[tadf_ps[i]]

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
