import torch
import torch.nn as nn
import torch.nn.functional as F

class CAModel(nn.Module):
    
    def __init__(self, env_d, hidden_d):
        super(CAModel, self).__init__()
        self.sc = 0
        self.lc = 32
        self.env_d = env_d
        self.hidden_d = hidden_d
        self.lA = self.make_lay()
        self.lB = self.make_lay()
        self.lC = self.make_lay()
        self.lD = self.make_lay()
        self.lE = self.make_lay()
        self.lF = self.make_lay()
        self.lG = self.make_lay()
        self.lH = self.make_lay()
        self.lI = self.make_lay()
        self.lJ = self.make_lay()
        self.lK = self.make_lay()
        self.lL = self.make_lay()
        self.lM = self.make_lay()
        self.lN = self.make_lay()
        self.lO = self.make_lay()
        self.lP = self.make_lay()
        self.lQ = self.make_lay()
        self.lR = self.make_lay()
        self.lS = self.make_lay()
        self.lT = self.make_lay()
        self.lU = self.make_lay()
        self.lV = self.make_lay()
        self.lW = self.make_lay()
        self.lX = self.make_lay()
        self.lY = self.make_lay()
        self.lZ = self.make_lay()
        self.lAA = self.make_lay()
        self.lBB = self.make_lay()
        self.lCC = self.make_lay()
        self.lDD = self.make_lay()
        self.lEE = self.make_lay()
        self.lFF = self.make_lay()
        self.lays = [
            self.lA, self.lB, self.lC, self.lD, self.lE, self.lF, self.lG, self.lH,
            self.lI, self.lJ, self.lK, self.lL, self.lM, self.lN, self.lO, self.lP,
            self.lQ, self.lR, self.lS, self.lT, self.lU, self.lV, self.lW, self.lX,
            self.lY, self.lZ, self.lAA, self.lBB, self.lCC, self.lDD, self.lEE, self.lFF
        ]

    def make_lay(self):
        conv1 = nn.Conv2d(self.env_d,self.hidden_d,3)
        conv2 = nn.Conv2d(self.hidden_d,self.env_d,3)
        #if (zeroed):
        #nn.init.zeros_(conv2.weight)
        #nn.init.zeros_(conv2.bias)
        return nn.Sequential(
            conv1,
            nn.ReLU(inplace=True),
            conv2
        )
        
    def forward(self, x):
        cv = self.lays[self.sc % self.lc]
        out = cv(x)
        self.sc += 1
        return out