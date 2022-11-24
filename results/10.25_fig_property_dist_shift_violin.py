import glob
import rdkit
from rdkit import Chem
import random
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import math 
import matplotlib
from matplotlib.ticker import MultipleLocator

random.seed(0)
fns = ['./total_QC_data.txt']

def violin_plot(datas, labels, colors,legend):
    plt.rc('font', size=15)
    fig, ax = plt.subplots()
    ps = [i for i in range(len(datas))]

    #values = ['50-60','60-70','70-80','80-90','90-100']
    values = [1,2,3,4,5]
    print(datas)
    violin = ax.violinplot(datas, positions=ps, widths=0.8)
    #ax.set_xticks(values)
    ax.set_xticks([i for i in range(len(labels))])
    ax.set_xlabel('TADF-likeness')
    ax.set_xticklabels(labels)
    ax.set_ylabel(f'$\Delta$ {legend}')
    for i, color in enumerate(colors):
        violin['bodies'][i].set_facecolor(color)

    violin['cbars'].set_edgecolor('gray')
    violin['cmaxes'].set_edgecolor('gray')
    violin['cmins'].set_edgecolor('gray')

    #y = [ np.mean(data) for data in datas]
    #plt.plot([i for i in range(len(labels))], y )
    #plt.legend()
    plt.show()

homo_c = -6.5
lumo_c = -1
s1_lc = 1.6
s1_uc = 3.1
est_c = 0.4

smiles_list = []

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
    
    pub_ps =  [obtain_mean_variance(ps) for ps in [pub_homo,pub_lumo, pub_s1, pub_st] ]
    return pub_ps

tadf_ps = read_data('../data/total_train_QC_data.txt')


homo_list = [[] for i in range(5)]
lumo_list = [[] for i in range(5)]
s1_list = [[] for i in range(5)]
est_list = [[] for i in range(5)]

for fn in fns:
    with open(fn) as f:
        lines = f.readlines()
    for line in lines:
        homo, lumo, s1, t1, f, score = [float(val) for val in line.split()[2:]]
        #print(homo, tadf_ps[0][0])                        
        homo = abs(homo-tadf_ps[0][0])
        lumo = abs(lumo-tadf_ps[1][0])
        est = abs(s1-t1 - tadf_ps[3][0])
        s1 = abs(s1-tadf_ps[2][0])
        i = int(score/10)-5
        homo_list[i].append(homo)
        lumo_list[i].append(lumo)
        s1_list[i].append(s1)
        est_list[i].append(est)
        
colors = ['y','orange','r','purple','black']


homo_list =  [homo_list[i]  for i in range(5)]
lumo_list =  [lumo_list[i]  for i in range(5)]
s1_list =  [s1_list[i]  for i in range(5)]
est_list = [est_list[i]  for i in range(5)]
print(est_list)

properties = [homo_list, lumo_list, s1_list, est_list]
#print(properties)
label_fontsize = 20
tick_length = 6
tick_width = 1.5
tick_labelsize = 16
legend_fontsize = 16


values = ['50-60','60-70','70-80','80-90','90-100']
labels = ['HOMO (eV)', 'LUMO (eV)', '$E(S_{1})$', '$\Delta E_{ST}$']
colors = ['y','orange','r','purple','black']

fig = plt.figure(figsize=(30,6))

for i in range(3,4):
    #plt.subplot(1,4,i+1)
    violin_plot(properties[i], values, colors,labels[i])
#plt.show()
