import torch
import torch.nn as nn
import torch.nn.functional as F

class CAModel(nn.Module):
    
    def __init__(self, env_d, hidden_d):
        super(CAModel, self).__init__()
        self.conv1 = nn.Conv2d(env_d*3,hidden_d,1)
        self.conv2 = nn.Conv2d(hidden_d,env_d,1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return self.conv2(x)