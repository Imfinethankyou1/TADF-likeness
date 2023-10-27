import torch
import torch.nn as nn

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
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
