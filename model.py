# Import the neural network architectures and libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
import random

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class DesEncoder(torch.nn.Module):

    def __init__(self, input_dim):
        super(DesEncoder, self).__init__()
        drop = 0.2
        self.rep_dim = int(input_dim*0.5*0.5*0.5*0.5)
        self.seq = nn.Sequential(nn.Linear(in_features=input_dim, out_features=int(input_dim*0.5)), nn.ReLU(), nn.Dropout(drop),
                 nn.Linear(in_features=int(input_dim*0.5), out_features=int(input_dim*0.5*0.5)), nn.ReLU(),nn.Dropout(drop),
                 nn.Linear(in_features=int(input_dim*0.5*0.5), out_features=5), nn.ReLU(),
                 #nn.Linear(in_features=int(input_dim*0.5*0.5*0.5), out_features=int(input_dim*0.5*0.5*0.5*0.5)), nn.ReLU()
                )
        
    def forward(self, x):
      return self.seq(x).squeeze()

class DesAutoEncoder(torch.nn.Module):

    def __init__(self, input_dim):
        super(DesAutoEncoder, self).__init__()
        self.encoder = DesEncoder(input_dim)
        self.encoder.apply(init_weights)
        output_dim = input_dim
        drop = 0.2
        self.decoder =  nn.Sequential(
                #nn.Linear(in_features=int(input_dim*0.5*0.5*0.5*0.5), out_features=int(output_dim*0.5*0.5*0.5)), nn.ReLU(),
                nn.Linear(in_features=5, out_features=int(output_dim*0.5*0.5)), nn.ReLU(),nn.Dropout(drop),
                nn.Linear(in_features=int(input_dim*0.5*0.5), out_features=int(output_dim*0.5)), nn.ReLU(),nn.Dropout(drop),
                nn.Linear(in_features=int(output_dim*0.5), out_features=output_dim), nn.Sigmoid())
                #nn.Linear(in_features=int(output_dim*0.5), out_features=output_dim))

        self.encoder.apply(init_weights)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AutoEncoder(nn.Module):
    #https://www.kaggle.com/code/rohitgr/autoencoders-tsne/notebook
    def __init__(self, f_in):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(f_in, 100),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(100, 70),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(70, 40)
        )
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(40, 70),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(70, f_in),
            #nn.Sigmoid()
            #nn.Hardsigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
