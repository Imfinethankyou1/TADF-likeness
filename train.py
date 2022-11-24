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
import pickle
from sklearn.preprocessing import MinMaxScaler
y_scaler1 = MinMaxScaler()

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
args.test_data = args.test_data.split()
args.vlabels = args.vlabels.split()
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
    for i in range(3,4):
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


plt.rc('font', size=10)
dataset = None
k_fold = 1
if args.train:
    for model_ind in range(k_fold):
        fns = []
        train_dataset = get_dataset_dataloader(args.tdata,split_ratio=0.9, batch_size=args.batch_size, num_workers=16, train=True)
        if args.vdata is not None:
            val_dataset = get_dataset_dataloader(args.vdata,split_ratio=0.9, batch_size=args.batch_size, num_workers=16, train=True)
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
    dataset = get_dataset_dataloader(args.tdata, split_ratio=0.9, batch_size=args.batch_size, num_workers=16, train=True)

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
unlabel_xs_list = []


test_fns = args.test_data

for test_fname in test_fns:
    uf_final = get_dataset_dataloader(test_fname,batch_size=args.batch_size, num_workers=16)
    unlabel_xs = uf_final[1]
    unlabel_xs_list.append(unlabel_xs)

lab_list = []
unlab_list = []
for model_ind in range(1):
    #Model load part    
    if True:
        save_model = args.save_model
        trainer.ae_net.load_state_dict(torch.load(save_model))
        trainer.ae_net.eval()
        
        lab = score(trainer, dataset[1]).cpu().detach().numpy()
        #lab= y_scaler1.fit_transform(lab.reshape(-1,1))
        lab = lab.reshape(-1)
        if len(lab_list) == 0:
            lab_list.append(lab)
        else:
            lab_list[0]+=lab
        print(f'train score : {np.mean(lab)}')

        print('make test')
        j = 0
        for unlabel_xs in unlabel_xs_list:
            unlab = score(trainer, unlabel_xs).cpu().detach().numpy()
            unlab = unlab.reshape(-1)
            if len(unlab_list) == j:
                unlab_list.append(unlab)
            else:
                unlab_list[j]+=unlab
            j+=1


lab = lab_list[0]
unlab_list = [unlab for unlab in unlab_list]
labels = args.vlabels

colors = ['y','r','lime','violet','g','orange','b','black','grey'][:len(labels)]

def calc_auroc(pos, neg):
    from sklearn.metrics import roc_auc_score,roc_curve
    import numpy as np

    true_list = np.array([1 for _ in range(len(pos))] + [0 for _ in range(len(neg))])
    score_list = np.array(pos + neg)

    return roc_auc_score(true_list, score_list)

#print('AUROC 0,1: ', calc_auroc(unlab_list[0].tolist(), unlab_list[1].tolist()) )
#print('AUROC 0,2: ', calc_auroc(unlab_list[0].tolist(), unlab_list[2].tolist()) )
print('test mean : ',[unlab.mean() for unlab in unlab_list])


remain_list =[ []  for i in range(len(labels))]
threshold_list = []
ratio_on = False
if ratio_on:
    for ratio in range(50):
        lab = lab[lab.argsort()]
        threshold = lab[int(ratio*len(lab)/100)]
        threshold_list.append(threshold)
        print(threshold, end = ' ')
        for j in range(len(unlab_list)):
            unlab = unlab_list[j]
            print(len(np.where(unlab>threshold)[0])/len(unlab), end =' ')
            remain_list[j].append( len(np.where(unlab>threshold)[0])/len(unlab) )
        print('lab : ',len(np.where(lab>threshold)[0])/ len(lab))

color_list = ['#feb308','#0165fc','black','y','r','lime','violet','g']
if False:


    import matplotlib
    from matplotlib.ticker import MultipleLocator
    label_fontsize = 20
    tick_length = 6
    tick_width = 1.5
    tick_labelsize = 16
    legend_fontsize = 16

    plt.ylim([0.0, 0.2])
    plt.xlim([0.0, 100.1])
    #plt.yticks([0,0.1,0.2,0.3])
    plt.yticks([0,0.05,0.1,0.15,0.2])
    plt.xticks([0,20,40,60,80,100])
    for i in range(len(args.vlabels)):
        sns.kdeplot(unlab_list[i], color=color_list[i], label=args.vlabels[i])

    #ax2.scatter(nnpp_r_cut_list, nnpp_norm_err_mean, marker='v', color='b')
    plt.xlabel(rf'TADF-likeness', fontsize=label_fontsize)
    plt.ylabel('Density', fontsize=label_fontsize, color='k')
    #ax2.set_ylabel('$\Delta N_{nc}$ (%)', fontsize=label_fontsize, color='b')
    plt.tick_params(length=tick_length, width=tick_width, labelsize=tick_labelsize, labelcolor='k', color='k')
    matplotlib.rcParams['ytick.major.pad'] = 3
    plt.tight_layout()
    plt.rc('font', size=40)
    plt.legend(prop={'size': 16}, loc='upper left', ncol=1)
    plt.show()

plot_on = True
if plot_on:
    j = 0
    datas = []
    for unlab in unlab_list:
        datas.append(unlab)
        j +=1
    violin_plot(datas, labels, colors)
