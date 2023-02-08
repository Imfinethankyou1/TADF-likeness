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
import pickle
from utils import score, set_cuda_visible_device
import random
import math

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('--batch_size', type=int, default=10000, help='Batch size for mini-batch training.')
parser.add_argument('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
parser.add_argument('--lr_decay', type=float, default=1.0,
              help='Lr decay hyperparameter .')
parser.add_argument('--lr', type=float, default=1e-3,
              help='Learning rate')
parser.add_argument('--epochs', type=int, default=10000,
              help='Epochs')
parser.add_argument('--objective', type=str, default='one-class',
              help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
parser.add_argument('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
parser.add_argument('--ae_optimizer_name', type=str, default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
parser.add_argument('--optimizer_name', type=str, default='adam',
              help='Name of the optimizer to use for Deep SVDD network training.')
parser.add_argument('--train', action='store_true',
              help='Train on/off')
parser.add_argument('--tdata', type=str, default=None, 
              help='Name of the training dataset')
parser.add_argument('--vdata', type=str, default=None, 
              help='Name of the validation dataset')
parser.add_argument('--test_data', type=str, default=None, 
              help='Name of the test dataset')
parser.add_argument('--vlabels', type=str, default=None, 
              help='Labels of the test dataset in Figure')
parser.add_argument('--save_model', type=str, default=None,
              help='Name of saved model parameters')
parser.add_argument('--pre_save_model', type=str, default=None,
              help='Name of saved pretrain model parameters')

args = parser.parse_args()
args.tdata = args.tdata.split()


args.tdata = [ '../TADF-AE/'+data  for data in args.tdata]
args.test_data = args.test_data.split()
args.vlabels = args.vlabels.split()
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

set_seed(0)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
cmd = set_cuda_visible_device(1)
os.environ['CUDA_VISIBLE_DEVICES'] = cmd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def violin_plot(datas, labels, colors):
    from matplotlib import pyplot as plt
    import seaborn as sns
    plt.rc('font', size=15)
    fig, ax = plt.subplots()
    ps = [i for i in range(len(datas))]

    violin = ax.violinplot(datas, positions=ps, widths=0.8)
    ax.set_xticks([i for i in range(len(labels))])
    ax.set_xlabel('Data Type')
    ax.set_xticklabels(labels)
    ax.set_ylabel('TADF-likeness')
    for i, color in enumerate(colors):
        violin['bodies'][i].set_facecolor(color)

    violin['cbars'].set_edgecolor('gray')
    violin['cmaxes'].set_edgecolor('gray')
    violin['cmins'].set_edgecolor('gray')

    y = [ np.mean(data) for data in datas]
    #plt.plot([i for i in range(len(labels))], y )

    plt.show()

def obtain_mean_variance(vals):
    mean = sum(vals) / len(vals)
    vsum = 0
    for val in vals:
        vsum = vsum + (val - mean)**2
    variance = vsum / len(vals)

    std = math.sqrt(variance)
    std = np.std(vals)
    return mean, std



plt.rc('font', size=10)
dataset = None
k_fold = 1
if args.train:
    for model_ind in range(k_fold):
        fns = []
        train_dataset = get_dataset_dataloader(args.tdata, batch_size=args.batch_size, num_workers=8, train=True)
        if args.vdata is not None:
            val_dataset = get_dataset_dataloader(args.vdata, batch_size=args.batch_size, num_workers=8, train=True)
            val_dataset = val_dataset[0]
        else:
            val_dataset = None
        X_train = train_dataset[1]


        print('num_descriptor : ',len(X_train[0]))
        dataset = train_dataset
        save_model = args.save_model
        #save_model = None
        autoencoder = AutoEncoder(len(X_train[0]))
        trainer = AETrainer(autoencoder,
                            [dataset[0], val_dataset],
                            optimizer_name=args.ae_optimizer_name,
                            lr=args.lr,
                            n_epochs = args.epochs,
                            lr_milestones=(),
                            batch_size=args.batch_size,
                            weight_decay=0.0,
                            save_model = save_model,
                            device = device,
                            lr_decay = args.lr_decay
                            )

        if not args.pre_save_model is None:
            print('load pretrain model')
            trainer.ae_net.load_state_dict(torch.load(args.pre_save_model))
        trainer.train()

if dataset is None:
    dataset = get_dataset_dataloader(args.tdata, batch_size=args.batch_size, num_workers=16, train=True)

autoencoder = AutoEncoder(200)
trainer = AETrainer(autoencoder,
                    None,
                    optimizer_name=args.ae_optimizer_name,
                    lr=1e-4,
                    n_epochs = 10000,
                    lr_milestones=(),
                    batch_size=args.batch_size,
                    weight_decay=0.0,
                    save_model = None,
                    device = device,
                    lr_decay = 1.0
                    )

#Model load part    
save_model = args.save_model
trainer.ae_net.load_state_dict(torch.load(save_model))
trainer.ae_net.eval()


#for x in dataset[1]:
#    print(x)
#print(np.mean(dataset[1]))
#sys.exit()

train_likeness = score(trainer, dataset[1], device).cpu().detach().numpy().reshape(-1)
print(f'train score : {np.mean(train_likeness)}')

mean, std = obtain_mean_variance(train_likeness)

print('train 1. mean 2. std : ', mean, std)

print('test start')
unlab_list = []
for test_fname in args.test_data:
    _, unlabel_xs, _ = get_dataset_dataloader(test_fname,batch_size=args.batch_size, num_workers=8)
        
    likeness_list = score(trainer, unlabel_xs, device).cpu().detach().numpy().reshape(-1)
    unlab_list.append(likeness_list)

colors = ['y','r','lime','violet','g','orange','b','black','grey'][:len(args.vlabels)]
violin_plot(unlab_list, args.vlabels, colors)
