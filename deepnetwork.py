import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
    def  __init__(self, size, l_rate=.1, zeroInit=False):
        super(CNN, self).__init__()
        fc_list = []
        for i in range(len(size)-1):
            fc = nn.Linear(size[i], size[i+1])
            fc_list.append(fc)
            if not zeroInit: torch.nn.init.xavier_uniform(fc.weight)
            else: torch.nn.init.constant_(fc.weight, 0.0)
        self.fc = nn.ModuleList(fc_list)

    def forward(self, x):
        for i, fc in enumerate(self.fc):
            x = fc(x)
            if i != len(self.fc)-1:
                x = F.relu(x)
        return x

