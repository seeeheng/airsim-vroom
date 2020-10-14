import torch

class QNetwork(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(qNetwork, self).__init__()
        self.adv1 = nn.Linear(state_size, 256)
        self.adv2 = nn.Linear(256, action_size)

        self.val1 = nn.Linear(state_size,256)
        self.val2 = nn.Linear(256, 1)

    def forward(self, state):
        adv = F.relu(self.adv1(state))
        adv = self.adv2(adv)

        val = F.relu(self.val1(state))
        val = self.val2(val)
        
        return val + adv - adv.mean()