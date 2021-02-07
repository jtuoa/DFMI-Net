import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


H=100 

class Mine(nn.Module):
    def __init__(self):
        super(Mine, self).__init__()        
        self.fc1 = nn.Linear(1, H)
        self.fc2 = nn.Linear(1, H)
        self.fc3 = nn.Linear(H, H)
        self.fc4 = nn.Linear(H, H)
        self.fc5 = nn.Linear(H, 1)


    def forward(self, x, y):
        h1 = F.relu(self.fc1(x)+self.fc2(y))
        h2 = self.fc3(h1)
        h3 = self.fc4(h2)
        h4 = self.fc5(h3)
        return h4
        
        
