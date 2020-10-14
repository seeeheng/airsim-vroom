import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(qNetwork, self).__init__()
        # Input is going to be 84x84x4
        self.conv1 = nn.Conv2d(3, 16, 8, stride=4) # 16 x 16 x 16 after pool
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2) # 8 x 8 x 32 after pool
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1) # should be 8 x 8 x 32

        # self.adv1 = nn.Linear(state_size, 256)
        self.adv1 = nn.Linear(256, 512)
        self.adv2 = nn.Linear(512, action_size)

        # self.val1 = nn.Linear(state_size,256)
        self.val1 = nn.Linear(256, 512)
        self.val2 = nn.Linear(512, 1)

    def forward(self, state):
        adv = F.relu(self.adv1(state))
        adv = self.adv2(adv)

        val = F.relu(self.val1(state))
        val = self.val2(val)
        
        return val + adv - adv.mean()