import numpy as np
import random
random.seed(0)
np.random.seed(seed=0)
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
#from sklearn.manifold.t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
import sys
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import *




if __name__ == '__main__':

    sns.set(rc={'figure.figsize':(11.7,8.27)})


    with open('./data/property_4.txt') as f:
        lines = f.readlines()

    data = np.load('./data/z/property_4.npz',allow_pickle=True)
    fp_list = data['feats']
    print(fp_list)
    label_list = [5 for i in range(len(lines))]
    test_num =50000
    for i in range(4,5):
        with open(f'../TADF-DeepSVDD/data_property/property_{i}.txt') as f:
            test_lines = f.readlines()
        lines += test_lines[:test_num]
        label_list += [i for k in range(len(test_lines))][:test_num]
        data = np.load(f'../TADF-DeepSVDD/data_property/z/property_{i}.npz',allow_pickle=True)
        fp_list +=data['feats'][:test_num]


    z_dim = len(fp_list[0])
    print(z_dim)
    y = []
    print('OK')
    dim_1 = len(lines)
    #dim_1 = 100
    x = np.zeros( (dim_1, z_dim))
    for i in range(dim_1):
        #for i in range(1000):
        y.append(int(label_list[i]))
        for j , val in enumerate(fp_list[i]):
            x[i][j] = float(val)
    #c_len = len(list(set(y)))
    c_len = 6
    palette = sns.color_palette()[:c_len]
    X = x
    print(X.shape)
    y = np.array(y)
    
    tsne = TSNE(n_components=2)
    X_embedded = tsne.fit_transform(X)

    train_ind_list=np.where(y==5)[0]
    test_ind_list = np.where(y!=5)[0]
    test_x = []
    test_y = []
    t_y = []
    for ind in test_ind_list:
        test_x.append(X_embedded[:,0][ind])
        test_y.append(X_embedded[:,1][ind])
        t_y.append(y[ind])

    train_x = []
    train_y = []
    tt_y = []
    for ind in train_ind_list:
        train_x.append(X_embedded[:,0][ind])
        train_y.append(X_embedded[:,1][ind])
        tt_y.append(y[ind])
    
    sns.scatterplot(np.array(test_x), test_y, s=100, hue=t_y, legend='full', palette=palette[:1], alpha=0.3)
    sns.scatterplot(train_x, train_y, s=100, hue=tt_y, legend='full',palette=[palette[5]])
    #sns.scatterplot(X_embedded[:,0], X_embedded[:,1], s=100, hue=y, legend='full',palette=palette[4])



    plt.show()

