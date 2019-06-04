import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self, net_parameters):
        super(Net, self).__init__()

        # network parameters
        IN_W, IN_H, IN_C, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F = net_parameters
        FC1_IN = CL2_F * (IN_W // 4)**2

        # Convolutional Layer 1
        PAD1 = (CL1_K - 1) // 2
        self.conv1 = nn.Conv2d(IN_C, CL1_F, CL1_K, padding=PAD1)
        Fin = IN_C * CL1_K**2
        Fout = CL1_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.conv1.weight.data.uniform_(-scale, scale)
        self.conv1.bias.data.fill_(0.0)

        # Convolutional Layer 2
        PAD2 = (CL2_K - 1) // 2
        self.conv2 = nn.Conv2d(CL1_F, CL2_F, CL2_K, padding=PAD2)
        Fin = CL1_F * CL2_K**2
        Fout = CL2_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.conv2.weight.data.uniform_(-scale, scale)
        self.conv2.bias.data.fill_(0.0)

        # Full Connected Layer 1
        self.fc1 = nn.Linear(FC1_IN, FC1_F)
        Fin = FC1_IN
        Fout = FC1_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.fc1.weight.data.uniform_(-scale, scale)
        self.fc1.bias.data.fill_(0.0)
        self.FC1_IN = FC1_IN

        # Full Connected Layer 2 
        self.fc2 = nn.Linear(FC1_F, FC2_F)
        Fin = FC1_F
        Fout = FC2_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.fc2.weight.data.uniform_(-scale, scale)
        self.fc2.bias.data.fill_(0.0)

        # Max Pooling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, prob):
        # Convolutional Layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Convolutional Layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Full Connected Layer 1
        x = x.view(-1, self.FC1_IN)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=prob, training=self.training)

        # Full Connected Layer 2
        x = self.fc2(x)
        
        return x


        
