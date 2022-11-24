import glob
import rdkit
from rdkit import Chem
import random
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import math 

random.seed(0)
fns = ['./total_QC_data.txt']



homo_c = -6.5
lumo_c = -1
s1_lc = 1.6
s1_uc = 3.1
est_c = 0.4

smiles_list = []


homo_list = [[] for i in range(5)]
lumo_list = [[] for i in range(5)]
s1_list = [[] for i in range(5)]
est_list = [[] for i in range(5)]

for fn in fns:
    with open(fn) as f:
        lines = f.readlines()
    for line in lines:
        homo, lumo, s1, t1, f, score = [float(val) for val in line.split()[2:]]

        i = int(score/10)-5
        homo_list[i].append(homo)
        lumo_list[i].append(lumo)
        s1_list[i].append(s1)
        est_list[i].append(s1-t1)

colors = ['y','orange','r','purple','black']

def obtain_mean_variance(vals):
    mean = sum(vals) / len(vals)
    vsum = 0
    for val in vals:
        vsum = vsum + (val - mean)**2
    variance = vsum / len(vals)

    std = math.sqrt(variance)
    std = np.std(vals)
    return [mean, std]

homo_list =  [obtain_mean_variance(homo_list[i])  for i in range(5)]
lumo_list =  [obtain_mean_variance(lumo_list[i])  for i in range(5)]
s1_list =  [obtain_mean_variance(s1_list[i])  for i in range(5)]
est_list = [obtain_mean_variance(est_list[i])  for i in range(5)]


properties = [homo_list, lumo_list, s1_list, est_list]
print(properties)
import matplotlib
from matplotlib.ticker import MultipleLocator
label_fontsize = 20
tick_length = 6
tick_width = 1.5
tick_labelsize = 16
legend_fontsize = 16



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
    
    pub_ps =  [obtain_mean_variance(ps) for ps in [pub_homo,pub_lumo, pub_s1, pub_st] ]
    return pub_ps

tadf_ps = read_data('../data/total_train_QC_data.txt')

values = ['50-60','60-70','70-80','80-90','90-100']
labels = ['HOMO (eV)', 'LUMO (eV)', '$E(S_{1})$', '$\Delta E_{ST}$']
colors = ['y','orange','r','purple','black']

fig = plt.figure(figsize=(15,12))

for i in range(4):
    plt.subplot(2,2,i+1)
    sample_props = []
    variances = []
    for j in range(5):
        sample_props.append(abs(properties[i][j][0]-tadf_ps[i][0]))
        variances.append(properties[i][j][1])

    s = [80  for k in range(len(variances))]
    #plt.errorbar(values, sample_props, yerr=variances, fmt="o", marker='^', lw=2, color=colors[i],capsize=5, capthick=2)   
    #plt.scatter(values, sample_props, marker='^', color=colors[i], label=labels[i], s= s)
    plt.bar(values, sample_props, color=colors[i], label=labels[i])
    plt.xlabel(rf'TADF-likeness', fontsize=label_fontsize)
    plt.ylabel('$\Delta$', fontsize=label_fontsize, color='k')
    plt.tick_params(length=tick_length, width=tick_width, labelsize=tick_labelsize, labelcolor='k', color='k')
    #matplotlib.rcParams['ytick.major.pad'] = 3
    #y_interval = round((max(sample_props)-min(sample_props))/5,1)
    #y_ticks = [ y_interval*k+round(min(sample_props),1) for k in range(1,5)]
    #plt.yticks(y_ticks)
    plt.legend(fontsize=legend_fontsize)
    #plt.tight_layout()
    #plt.rc('font', size=10)
#label = 'MAE of eigenvalues (eV)'
plt.show()
