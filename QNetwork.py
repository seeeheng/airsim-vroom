import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # Input is going to be 84x84x4
        self.conv1 = nn.Conv2d(4, 16, 8, stride=4) # 16x20x20 after pool
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2) # 32x9x9 after pool
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1) # 32x7x7

        # self.adv1 = nn.Linear(state_size, 256)
        self.adv1 = nn.Linear(1568, 512)
        self.adv2 = nn.Linear(512, action_size)

        # self.val1 = nn.Linear(state_size,256)
        self.val1 = nn.Linear(1568, 512)
        self.val2 = nn.Linear(512, 1)

    def forward(self, state):
        # x = x.view(-1, 64*4*4) # this will flatten the layers why??
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1,x.shape[0],32*7*7)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)
        
        return val + adv - adv.mean()