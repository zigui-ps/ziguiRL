import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
    def  __init__(self, size, l_rate=.1, tanh=False, scale = 1, offset = 0):
        super(CNN, self).__init__()
        fc_list = []
        for i in range(len(size)-1):
            fc = nn.Linear(size[i], size[i+1])
            fc_list.append(fc)
            #torch.nn.init.xavier_uniform(fc.weight)
        self.fc = nn.ModuleList(fc_list)
        self.fc[-1].weight.data.mul_(0.1)
        self.fc[-1].bias.data.mul_(0.0)
        self._tanh = tanh
        self._scale = scale
        self._offset = offset

    def forward(self, x):
        for i, fc in enumerate(self.fc):
            x = fc(x)
            if i != len(self.fc)-1:
                x = F.relu(x)
        if self._tanh: x = F.tanh(x)
        x = (x + self._offset) * self._scale
        return x

