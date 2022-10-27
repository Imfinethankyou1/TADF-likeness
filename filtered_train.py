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
import sys
paths = ['../models/python_codes/', '../models/deep_one_class/', '../models/deep_one_class/src']
sys.path.extend(paths)
from deep_model import *

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 
torch.manual_seed(0)
deepSVDD.build_network = build_network
deepSVDD.build_autoencoder = build_autoencoder

dataset_name = args.dataset_name
normal_class = args.normal_class
net_name = args.net_name
n_jobs_dataloader = 0
dataset = get_dataset_dataloader('data/filtered_total_TADF.txt',batch_size=args.batch_size, num_workers=10)
#dataset.to(device)

unlabel_xs_list = []
for test_fname in ['data/filtered_test_TADF.txt','data/filtered_Pub_data.txt','data/filtered_test_gdb.txt', 'data/filtered_test_clean.txt','data/filtered_random_low_simil_10_5_pub.npz' ]:
#for test_fname in ['data/filtered_test_TADF.txt' ]:
#for test_fname in ['data/model_test.txt']:
    uf_final = get_dataset_dataloader(test_fname,batch_size=args.batch_size, num_workers=10)
    unlabel_xs = uf_final[4]
    uf_final = uf_final[2]
    unlabel_xs_list.append(unlabel_xs)

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

set_seed(0)
deep_SVDD = deepSVDD.DeepSVDD(args.objective, args.nu)
deep_SVDD.set_network(net_name)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Perform 30 runs for measuring the mean correlation and mean standard deviation of the scores between the 30 different runs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def score(deep_SVDD, X):
    with torch.no_grad():
        net = deep_SVDD.net.to(device)
        X = torch.FloatTensor(X).to(device)
        y = net(X)
        c, R = torch.FloatTensor([deep_SVDD.c]).to(device), torch.FloatTensor([deep_SVDD.R]).to(device)
        dist = torch.sum((y - c)**2, dim=1)
        if deep_SVDD.objective == 'soft-boundary':
            scores = dist - R ** 2
        else:
            scores = dist
    return scores

lab_list=[]
unlab_list=[]
epochs = 10
plt.rc('font', size=40)

print('Have to make k-fold closs-validation method')
print('Bayesian OTP process have to be implemeted')

#for i in range(30):
for i in range(1):
    save_model = f'trained_model/save_{i}.pt'
    set_seed(i)
    X_train = dataset[3]

    deep_SVDD = deepSVDD.DeepSVDD(args.objective, args.nu)
    deep_SVDD.set_network(net_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                      
    deep_SVDD.pretrain(dataset, optimizer_name=args.ae_optimizer_name,
                     lr=1e-5,
                     n_epochs = epochs ,
                     #n_epochs = 1,
                     lr_milestones=(100,),
                     batch_size=200, 
                     weight_decay=0.5e-3,  
                     device=device,
                     n_jobs_dataloader=0)
    deep_SVDD.train(dataset,
                  optimizer_name=args.optimizer_name,
                  lr=1e-5,
                  n_epochs = epochs,
                  #lr_milestones=args.lr_milestone,
                  lr_milestones=(1000,),
                  batch_size=args.batch_size,
                  weight_decay=args.weight_decay,
                  device=device,
                n_jobs_dataloader=n_jobs_dataloader)
    torch.save(deep_SVDD, save_model)
    #Model load part
    #model = torch.load(save_model)
    #model.eval()

    y_scaler1 = MinMaxScaler()
    lab = score(deep_SVDD, X_train).cpu().detach().numpy()*-1 
    lab_list.append(lab.ravel())
    lab= y_scaler1.fit_transform(lab.reshape(-1,1))
    print('train mean socre : ', np.mean(lab), np.min(lab))
    lab = lab.reshape(-1)
    threshold = np.mean(lab)
    sns.kdeplot(lab, shade=True, label='TADF',color='b')
    j = 0
    print('TADF treshold : ',threshold)
    labels = ['test_TADF','Pub', 'GDB-17','Clean','low_simil']
    colors = ['y','r','lime','violet','g']
    datas = [lab]
    for unlabel_xs in unlabel_xs_list:
        unlab = score(deep_SVDD, unlabel_xs).cpu().detach().numpy()*-1
        unlab= y_scaler1.fit_transform(unlab.reshape(-1,1))
        unlab_list.append(unlab.ravel())
        print('unlabel mean score : ', np.mean(unlab))

        unlab = unlab.reshape(-1)
        print(unlab.shape)
        print(unlab)
        #for k in range(len(unlab)):
        #    if unlab
        print(f'total num upper threshold 1.{labels[j]} 2.TADF: ', len(np.where(unlab>threshold)[0]), len(unlab), len(np.where(lab>threshold)[0]), len(lab) )
        sns.kdeplot(unlab, shade=True, label=labels[j],color=colors[j])
        j +=1
        datas.append(unlab)
    
    #fig, ax = plt.subplots()
    #ps = [i for i in range(len(datas))]
    #violin = ax.violinplot(datas, positions=ps)
    #ax.set_xlabel('Data Type')
    #ax.set_ylabel('Distribution')

    #violin['cbars'].set_edgecolor('gray')
    #violin['cmaxes'].set_edgecolor('gray')
    #violin['cmins'].set_edgecolor('gray')
    #plt.show()

    plt.legend()
    plt.show()
    # caclulate the Pearson correlattion and the Standard deviation of the scores after 30 runs, with 90% of the labelled data used as the training and 10% as the validation set
    sys.exit()
