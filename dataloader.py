import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
#from descriptastorus.descriptors import rdNormalizedDescriptors
import pickle


def get_dataset_dataloader(fn, batch_size=8, shuffle=True,
                           num_workers=1, length=None, train=False):
    dataset = DescriptorDataset(fn, length)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=shuffle, pin_memory=True)
    
    X_train = []
    key_list = []
    train_idx_list = []        
    for train_feature_dict in dataloader:
        X_train += train_feature_dict['fp'].tolist()
        key_list += train_feature_dict['key']
        train_idx_list+=train_feature_dict['key']

    return dataloader, X_train, key_list


class DescriptorDataset(Dataset):

    def __init__(self, fn, length=None):
        if isinstance(fn,list):
            fns = fn
            total_data = {'id':[], 'prop':[], 'feats':[], 'smiles':[]}
            for fn in fns:
                fn = fn.strip()[:-3]+'npz'
                print(fn)
                data = np.load(fn,allow_pickle=True)
                total_data['id']+=data['id']
                total_data['prop']+=data['prop']
                total_data['feats']+=data['feats']
                total_data['smiles'] += data['smiles']
            self.key = total_data['id']
            self.target = total_data['prop']
            fp_list = total_data['feats']
            smiles_list = total_data['smiles']

        else:
            fn = fn.strip()[:-3]+'npz'
            print(fn)
            data = np.load(fn,allow_pickle=True)
            self.key = data['id']
            self.target = data['prop']
            fp_list = data['feats']
            smiles_list = data['smiles']
            
        self.fp_list = []
        self.smiles_list = smiles_list
        assert len(fp_list) == len(smiles_list)
        for fp in fp_list:
            fp = np.array(fp)
            fp[np.isnan(fp)] = 0
            self.fp_list.append(fp)


    def __getitem__(self, idx):

        feature_dict = dict()

        feature_dict["key"] = self.key[idx]
        feature_dict["target"] = 1
        feature_dict["fp"] = self.fp_list[idx]
        feature_dict['smiles'] =self.smiles_list[idx]
        return feature_dict

    def __len__(self):
        return len(self.key)


if __name__ == "__main__":
    fn='../data/test.txt'

    dataset, data_loader = get_dataset_dataloader(
        fn, batch_size=4, num_workers=1)
    data_iter = iter(data_loader)
    for idx,batch in enumerate(data_loader):
        print(batch['fp'])

