import numpy as np
import argparse
from dataloader import *
import torch
import sys
from model import *
import os
from ae_trainer import AETrainer
from utils import score, set_cuda_visible_device

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('--dataset_name',type=str,help='dataset name', default=None)
parser.add_argument('--output_name',type=str,help='output name', default=None)
parser.add_argument('--batch_size', type=int, default=100000, help='Batch size for mini-batch training.')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
cmd = set_cuda_visible_device(1)
os.environ['CUDA_VISIBLE_DEVICES'] = cmd
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)


save_model = './trained_model/TADF_final.pt'
autoencoder = AutoEncoder(200)
autoencoder.to(device)
trainer = AETrainer(autoencoder,
                    None,
                    optimizer_name=None,
                    lr=1e-3,
                    n_epochs = 1,
                    lr_milestones=(),
                    batch_size=args.batch_size,
                    weight_decay=0.0,
                    save_model = save_model,
                    device = device,
                    lr_decay = 1.0
                    )

_, unlabel_xs, key_list = get_dataset_dataloader(args.dataset_name,batch_size=args.batch_size, num_workers=8)
trainer.ae_net.load_state_dict(torch.load(save_model))
trainer.ae_net.eval()

unlab = score(trainer, unlabel_xs, device).cpu().detach().numpy().reshape(-1)

with open(args.dataset_name.replace('origin/','')) as f:
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

with open(f'results/{args.output_name}','w') as f:
    for line in new_key_list:
        f.write(line+f' {line2likeness[line]}\n')
