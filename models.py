import torch
import torch.nn as nn
import torch.nn.functional as F


class DQModel(nn.Module):

    def __init__(self, state_dim, action_size):
        super(DQModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.bn4(self.fc3(x)))
        return self.fc5(x).to(device)
        
    
# Dueling DQN (aka DDQN)
class DDQNModel(nn.Module):
    
    def __init__(self, state_dim, action_size):
        super(DDQNModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 64)
        self.bn4 = nn.BatchNorm1d(64)
        
        # state_value stream
        self.state_value_fc = nn.Linear(64, 128)
        self.state_value_bn = nn.BatchNorm1d(128)
        self.state_value_out = nn.Linear(128, 1)
        
        # A(s, a) stream
        self.adv_fc = nn.Linear(64, 128)
        self.adv_bn = nn.BatchNorm1d(128)
        self.adv_out = nn.Linear(128, action_size)
        

    def forward(self, x):
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.bn4(self.fc3(x)))
        value = self.state_value_out(self.state_value_bn(self.state_value_fc(x)))
        adv = self.adv_out(self.adv_bn(self.adv_fc(x)))
        output = value + (adv - torch.mean(adv, dim=1, keepdim=True))
        return output
