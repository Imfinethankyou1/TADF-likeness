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


def get_dataset_dataloader(fn, split_ratio=0.9,batch_size=8, shuffle=True,
                           num_workers=1, length=None):
    dataset = DescriptorDataset(fn, length)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=shuffle, pin_memory=True)
    
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=shuffle, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=shuffle, pin_memory=True)
    
    end = True
    X_train = []
    key_list = []
    i =0
    train_data_iter = iter(train_dataloader) 
    train_feature_dict= next(train_data_iter)
    while end: 
        X_train += train_feature_dict['fp'].tolist()
        key_list += train_feature_dict['key']
        i+=1
        train_feature_dict = next(train_data_iter, None)
        if train_feature_dict is None:
            end =False
    X_total = X_train[:]
    X_test = []
    test_data_iter = iter(test_dataloader) 
    test_feature_dict= next(test_data_iter)
    end = True
    while end: 
        X_total += test_feature_dict['fp'].tolist()
        X_test += test_feature_dict['fp'].tolist()
        key_list += test_feature_dict['key']
        i+=1
        test_feature_dict = next(test_data_iter, None)
        if test_feature_dict is None:
            print('?')
            end =False

    print(len(X_train))
    return train_dataloader, test_dataloader, dataset,X_train, X_total, key_list, X_test


class DescriptorDataset(Dataset):

    def __init__(self, fn, length=None):
        if isinstance(fn,list):
            fns = fn
            total_data = {'id':[], 'prop':[], 'feats':[]}
            for fn in fns:
                fn = fn.strip()[:-3]+'npz'
                print(fn)
                data = np.load(fn,allow_pickle=True)
                total_data['id']+=data['id']
                total_data['prop']+=data['prop']
                total_data['feats']+=data['feats']
            self.key = total_data['id']
            self.target = total_data['prop']
            fp_list = total_data['feats']

        else:
            fn = fn.strip()[:-3]+'npz'
            print(fn)
            data = np.load(fn,allow_pickle=True)
            self.key = data['id']
            self.target = data['prop']
            fp_list = data['feats']
        self.fp_list = []
        
        for fp in fp_list:
            fp = np.array(fp)
            fp[np.isnan(fp)] = 0
            self.fp_list.append(fp)


    def __getitem__(self, idx):

        feature_dict = dict()

        feature_dict["key"] = self.key[idx]
        feature_dict["target"] = 1
        feature_dict["fp"] = self.fp_list[idx]
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

