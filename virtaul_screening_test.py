# Import the basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm
import itertools
from scipy.spatial.distance import squareform
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import argparse
from dataloader import *
import torch
import seaborn as sns
#from train import score
import sys
from model import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sys import argv
import os
import subprocess
from ae_trainer import AETrainer

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('--dataset_name',type=str,help='dataset name', default='123')
parser.add_argument('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
parser.add_argument('--net_name', type=str, help='neural Net name')
parser.add_argument('--batch_size', type=int, default=10000, help='Batch size for mini-batch training.')
parser.add_argument('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
parser.add_argument('--objective', type=str, default='one-class',
              help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
parser.add_argument('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
parser.add_argument('--ae_optimizer_name', type=str, default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
parser.add_argument('--optimizer_name', type=str, default='adam',
              help='Name of the optimizer to use for Deep SVDD network training.')
parser.add_argument('--lr_milestone', type=float, default=0, 
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')

args = parser.parse_args()

def score(trainer, X):
    net = trainer.ae_net
    with torch.no_grad():
        X = torch.FloatTensor(X).to(device)
        y = net(X)
        dist = torch.sum((y - X)**2, dim=1)
        scores = 100-4*dist
    return scores

def set_cuda_visible_device(ngpus):
    empty = []
    for i in range(8):
        command = ['nvidia-smi','-i',str(i)]
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        result = str(p.communicate()[0])
        count = result.count('No running')
        if count>0:
            empty.append(i)
    if len(empty)<ngpus:
        assert False, f"Available gpus are less than required: ngpus={ngpus}, empty={len(empty)}"
    cmd = ''
    for i in range(ngpus):
        cmd += str(empty[i]) + ','
    return cmd.rstrip(',')

def calc_auroc(pos, neg):
    from sklearn.metrics import roc_auc_score,roc_curve
    import numpy as np

    true_list = np.array([1 for _ in range(len(pos))] + [0 for _ in range(len(neg))])
    score_list = np.array(pos + neg)

    return roc_auc_score(true_list, score_list)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
cmd = set_cuda_visible_device(1)
#if not args.disable_cuda:
os.environ['CUDA_VISIBLE_DEVICES'] = cmd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 
torch.manual_seed(0)


n_jobs_dataloader = 0
dataset = get_dataset_dataloader('data_4/origin/property_0.txt',split_ratio=0.9, batch_size=args.batch_size, num_workers=16)

save_model = './trained_model/TADF_dopant.pt'
autoencoder = AutoEncoder(len(dataset[1][0]))
autoencoder.to(device)
trainer = AETrainer(autoencoder,
                    [dataset[0], dataset[0]],
                    optimizer_name=None,
                    lr=1e-3,
                    n_epochs = 1,
                    lr_milestones=(),
                    batch_size=100000,
                    weight_decay=0.0,
                    save_model = save_model,
                    device = device,
                    lr_decay = 1.0
                    )
#threshold = 96.55591 # prop_5
#threshold = 98.571724 # exp
threshold = 95.781044 # exp, pretrain
data_list = []
for test_fname in ['data/sample/random_2_log5.txt']:
    uf_final = get_dataset_dataloader(test_fname,batch_size=args.batch_size, num_workers=16)
    trainer.ae_net.load_state_dict(torch.load(save_model))
    #print('make test')

    unlabel_xs = uf_final[1]
    key_list = uf_final[-1]


    unlab = score(trainer, unlabel_xs).cpu().detach().numpy()
    unlab = unlab.reshape(-1)
    data_list.append(unlab)
    #sns.kdeplot(unlab,shade=True)
    #plt.show()
    ind_list = set(list(np.where(unlab>threshold)[0]))
    print('select : ',len(ind_list))
#with open(test_fname.replace('origin/','')) as f:
with open(test_fname.replace('sample/','')) as f:
    lines = f.readlines()
ind2line = {}
set_key_list = set(key_list)

for line in lines:
    if line.split()[0] in set_key_list:
        ind2line[line.split()[0]]=line.strip()
line2likeness = {}
new_key_list = []
for i in range(len(key_list)):
    if i in ind_list:
        new_key = ind2line[key_list[i]]
        line2likeness[new_key] = unlab[i]
        new_key_list.append(new_key)
    else:
        new_key = ind2line[key_list[i]]
        print('not : ', new_key)
new_key_list.sort(key=lambda x : line2likeness[x],reverse=True)

with open(f'data/sample/TADF_from_pubchem.txt','w') as f:
    for line in new_key_list:
        f.write(line+f' {line2likeness[line]}\n')

#fpr, tpr, thresholds = roc_curve(true_list, score_list, pos_label=1)

#plt.plot(fpr,tpr)
#plt.show()

