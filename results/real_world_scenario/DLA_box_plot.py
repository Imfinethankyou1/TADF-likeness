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
lumo_c = -0.85
s1_lc = 1.6
s1_uc = 3.1
est_c = 0.4

smiles_list = []

total_num = 3

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


fns = ['medium_QC_data.txt','high_QC_data.txt']

for idx, fn in enumerate(fns):
    with open(fn) as f:
        lines = f.readlines()
    for line in lines:
        homo, lumo, s1, t1, f, score = [float(val) for val in line.split()[2:]]
        est = s1-t1

        homo_list[idx].append(homo)
        lumo_list[idx].append(lumo)
        s1_list[idx].append(s1)
        est_list[idx].append(est)
        lines_list[idx].append(line)

#sys.exit()
colors = ['y','orange','r','purple','black']


homo_list =  [homo_list[i]  for i in range(total_num)]
lumo_list =  [lumo_list[i]  for i in range(total_num)]
s1_list =  [s1_list[i]  for i in range(total_num)]
est_list = [est_list[i]  for i in range(total_num)]


print('80-100 len : ', len(homo_list[-1]))
properties = [homo_list, lumo_list, s1_list, est_list]

print(est_list[-1])
#print(properties)
import matplotlib
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as ticker

label_fontsize = 20
tick_length = 6
tick_width = 1.5
tick_labelsize = 16
legend_fontsize = 16




#values = ['0-20','20-40','40-60','60-80','80-100']

values =  ['Low', 'Medium', 'High'] + ['TADF']
labels = ['HOMO (eV)', 'LUMO (eV)', '$E(S_{1})$', '$\Delta E_{ST}$']
colors = ['y','orange','r','purple','black']
ylim_list = [[-6.7,-4.1], [-3.0,-0.5], [1.7, 4.7], [-0.02, 1.6] ]

#fig = plt.figure(figsize=(30,6))
fig = plt.figure(figsize=(15,12))


for i in range(4):
    ax =plt.subplot(2,2,i+1)
    #sample_props = []
    sample_props = properties[i]+[tadf_ps[i]]
    #variances = []
    #for j in range(5):
    #    sample_props.append(abs(properties[i][j][0]-tadf_ps[i][0]))
    #    variances.append(properties[i][j][1])

    #s = [80  for k in range(len(variances))]
    #plt.errorbar(values, sample_props, yerr=variances, fmt="o", marker='^', lw=2, color=colors[i],capsize=5, capthick=2) 
    #plt.scatter(values, sample_props, marker='^', color=colors[i], label=labels[i], s= s)
    #data = [np.random.normal(0, std, 1000) for std in range(1, 6)]

    box = plt.boxplot(sample_props[1:], notch=False, patch_artist=True, labels=values[1:])
    colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink','ivory'][2:]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    #plt.plot([1,total_num],[tadf_ps[i][0] for k in range(2)])
    #plt.bar(values, sample_props, color=colors[i], label=labels[i])
    plt.xlabel('TADF-likeness', fontsize=label_fontsize)
    plt.ylabel(f'{labels[i]}', fontsize=label_fontsize, color='k')
    plt.tick_params(length=tick_length, width=tick_width, labelsize=tick_labelsize, labelcolor='k', color='k')
    plt.ylim(ylim_list[i])
    ax.yaxis.set_major_locator(ticker.LinearLocator(5)) 
    #matplotlib.rcParams['ytick.major.pad'] = 3
    #y_interval = round((max(sample_props)-min(sample_props))/5,1)
    #y_ticks = [ y_interval*k+round(min(sample_props),1) for k in range(1,5)]
    #plt.yticks(y_ticks)
    #plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    #plt.rc('font', size=10)
#label = 'MAE of eigenvalues (eV)'
#plt.subplots(constrained_layout=True)
plt.show()
