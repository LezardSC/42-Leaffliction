import torch.nn as nn
import torch.nn.functional as F


# Define a convolution neural network
# Given that the kernel size is 5
# We lose one pixel on each side going through each convolutional layer
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=10,
                               kernel_size=5,
                               stride=1,
                               padding=1)
        self.gn1 = nn.GroupNorm(num_groups=self._choose_groups(10),
                                num_channels=10)
        self.conv2 = nn.Conv2d(in_channels=10,
                               out_channels=15,
                               kernel_size=5,
                               stride=1,
                               padding=1)
        self.gn2 = nn.GroupNorm(num_groups=self._choose_groups(15),
                                num_channels=15)
        self.fc1 = nn.Linear(15*252*252, 4)

    def forward(self, input):
        output = F.relu(self.gn1(self.conv1(input)))
        output = F.relu(self.gn2(self.conv2(output)))
        flatten = nn.Flatten(start_dim=1)
        output = flatten(output)
        output = self.fc1(output)

        return output

    def _choose_groups(self, num_channels: int) -> int:
        for g in [32, 16, 8, 4, 2, 1]:
            if num_channels % g == 0:
                return g
        return 1
