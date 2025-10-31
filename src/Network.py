import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

# Define a convolution neural network
# Given that the kernel size is 5, we lose one pixel on each side of the picture going through each convolutional layer
class Network(nn.Module):
    def __init__(self, batch_size):
        super(Network, self).__init__()
        self.batch_size = batch_size
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(15)
        self.fc1 = nn.Linear(batch_size*15*252*252, 4)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = output.view(-1, self.batch_size*15*252*252)
        output = self.fc1(output)

        return output