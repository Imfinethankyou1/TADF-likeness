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
    for i in range(4):
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

save_model = './trained_model/TADF.pt'
#save_model = 'trained_model/prop_4_sig.pt'
autoencoder = AutoEncoder(200)
autoencoder.to(device)
trainer = AETrainer(autoencoder,
                    None,
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


data_list = []

with open('../TADF-DeepSVDD/data_rm_duple/all_pub/label.txt') as f:
    lines = f.readlines()

ind2txt = {}
for line in lines:
    ind, txt = line.strip().split()
    ind2txt[ind] = txt


for ind in range(int(sys.argv[1]),int(sys.argv[1])+10):
    for test_fname in [f'../TADF-DeepSVDD/data_rm_duple/all_pub/{ind}.txt']:
        uf_final = get_dataset_dataloader(test_fname,batch_size=100000, num_workers=4)
        trainer.ae_net.load_state_dict(torch.load(save_model))

        unlabel_xs = uf_final[1]
        key_list = uf_final[-1]


        unlab = score(trainer, unlabel_xs).cpu().detach().numpy()
        unlab = unlab.reshape(-1)
        data_list.append(unlab)

    #ind = test_fname.split('/')[-1].split('.txt')[0]
    ind = str(ind)

    with open(ind2txt[ind] ) as f:
        lines = f.readlines()
    ind2line = {}
    set_key_list = set(key_list)

    for line in lines:
        if line.split()[0] in set_key_list:
            ind2line[line.split()[0]]=line.strip()
    line2likeness = {}
    new_key_list = []
    for i in range(len(key_list)):
        new_key = ind2line[key_list[i]]
        line2likeness[new_key] = unlab[i]
        new_key_list.append(new_key)

    new_key_list.sort(key=lambda x : line2likeness[x],reverse=True)

    with open(f'data/sample/pubchem_sampling/{ind}.txt','w') as f:
        for line in new_key_list:
            if float(line2likeness[line]) < 90:
                break
            f.write(line+f' {line2likeness[line]}\n')


