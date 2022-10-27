# Import the basic libraries
import numpy as np
import matplotlib.pyplot as plt 
import argparse
from dataloader import *
import torch
import seaborn as sns
import sys
from model import *
from ae_trainer import AETrainer
import os
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser(description='Input')
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

args = parser.parse_args()

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

set_seed(0)

def score(trainer, X):
    net = trainer.ae_net
    with torch.no_grad():
        X = torch.FloatTensor(X).to(device)
        y = net(X)
        dist = torch.sum((y - X)**2, dim=1)
        scores = 100-4*dist
    return scores

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set gpu option
def set_cuda_visible_device(ngpus):
    empty = []
    for i in range(2):
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
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
cmd = set_cuda_visible_device(1)
os.environ['CUDA_VISIBLE_DEVICES'] = cmd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def violin_plot(datas, labels, colors):
    from matplotlib import pyplot as plt
    import seaborn as sns
    plt.rc('font', size=40)
    fig, ax = plt.subplots()
    ps = [i for i in range(len(datas))]

    violin = ax.violinplot(datas, positions=ps)
    ax.set_xticks([i for i in range(len(labels))])
    ax.set_xlabel('Data Type')
    ax.set_xticklabels(labels)
    ax.set_ylabel('Distribution')
    for i, color in enumerate(colors):
        violin['bodies'][i].set_facecolor(color)

    violin['cbars'].set_edgecolor('gray')
    violin['cmaxes'].set_edgecolor('gray')
    violin['cmins'].set_edgecolor('gray')
    plt.show()

#if torch.cuda.is_available():
dataset = get_dataset_dataloader('../TADF-DeepSVDD/data_property/origin/property_5.txt',split_ratio=0.9, batch_size=args.batch_size, num_workers=16, device= device)

X_train = dataset[3]
X_test = dataset[-1]
print('num_descriptor : ',len(X_train[0]))

label_test_xs_list = dataset[4]


lab_list=[]
unlab_list=[]
plt.rc('font', size=10)

train = True
for epochs in [10000]:
    save_model = f'trained_model/origin_save_0.pt'
    autoencoder = DesAutoEncoder(len(X_train[0]))
    autoencoder.to(device)
    trainer = AETrainer(autoencoder,
                        dataset,
                        optimizer_name=args.ae_optimizer_name,
                        lr=1e-3,
                        n_epochs = epochs,
                        lr_milestones=(),
                        batch_size=args.batch_size,
                        weight_decay=0.0,
                        save_model = save_model,
                        device = device,
                        lr_decay = 0.999
                        )
    pretrain = False
    if pretrain:
        #pretrain
        pre_dataset = get_dataset_dataloader('../TADF-DeepSVDD/data_rm_duple/all_pub/46.txt', split_ratio=0.9, batch_size=args.batch_size, num_workers=16, device= device )
        trainer = AETrainer(autoencoder,
                            pre_dataset,
                            optimizer_name=args.ae_optimizer_name,
                            lr=1e-3,
                            n_epochs = 100,
                            lr_milestones=(),
                            batch_size=args.batch_size,
                            weight_decay=0.0,
                            save_model = 'trained_model/pre_origin_save_0.pt',
                            device = device
                            )

        trainer.train()
        sys.exit()
    if train:
        #trainer.ae_net.load_state_dict(torch.load('trained_model/pre_origin_save_0.pt'))
        trainer.datatset = dataset
        trainer.train()
        
    #Model load part
    if True:
        #y_scaler = MinMaxScaler()
        trainer.ae_net.load_state_dict(torch.load(save_model))
        lab = score(trainer, X_train).cpu().detach().numpy()
        #lab= y_scaler.fit_transform(lab.reshape(-1,1))
        lab = lab.reshape(-1)
        y_true = [1 for i in range(len(label_test_xs_list))]
        #print('train mean score : ', np.mean(lab), np.min(lab))
        lab = lab[lab.argsort()]
        lab_test = score(trainer, X_test).cpu().detach().numpy()
        print(f'train score : {np.mean(lab)} test score : {np.mean(lab_test)}')

        print('make test')
        unlabel_xs_list = []
        unlabel_key_list = []
        for test_fname in ['data_property/origin/property_0.txt', 'data_property/origin/property_1.txt',
                           'data_property/origin/property_2.txt',
                           'data_property/origin/property_3.txt', 'data_property/origin/property_4.txt']:
            test_fname = '../TADF-DeepSVDD/'+test_fname
            uf_final = get_dataset_dataloader(test_fname,batch_size=args.batch_size, num_workers=16)
            unlabel_xs = uf_final[4]
            key_list = uf_final[5]
            unlabel_xs_list.append(unlabel_xs)
            unlabel_key_list.append(key_list)
        labels = ['0','1', '2','3','4','5']
        colors = ['y','r','lime','violet','g','orange']
        ratio_on = False
        if ratio_on:
            for ratio in [i for i in range(10)]:
                j = 0
                threshold = ratio*0.1 
                print(threshold)
                for unlabel_xs, key_list in zip(unlabel_xs_list, unlabel_key_list):
                    unlab = score(trainer, unlabel_xs).cpu().detach().numpy()
                    unlab = unlab.reshape(-1)
                    print(f'{labels[j]} : ', len(np.where(unlab>threshold)[0]), len(unlab), len(np.where(lab>threshold)[0]), len(lab), len(np.where(lab_test>threshold)[0])  ,len(lab_test) )

        plot_on = True
        if plot_on:
            j = 0
            datas = []
            for unlabel_xs, key_list in zip(unlabel_xs_list, unlabel_key_list):
                unlab = score(trainer, unlabel_xs).cpu().detach().numpy()
                unlab = unlab.reshape(-1)
                print(labels[j] , np.mean(unlab))
                datas.append(unlab)
                j +=1
            datas.append(lab)
            violin_plot(datas, labels, colors)
