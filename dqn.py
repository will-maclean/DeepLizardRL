import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, img_height, img_width, output_size=2):
        super().__init__()

        # Unfortunately this is the biggest nn I can put in my laptop without it having an asthma attack, feel free
        # to put a bigger nn in and update the forward method appropriately

        self.fc1 = nn.Linear(in_features=3*img_width*img_height, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=10)
        self.out = nn.Linear(in_features=10, out_features=output_size)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t
