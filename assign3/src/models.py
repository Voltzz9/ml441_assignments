# imports
import torch
import torch.nn as nn
import torch.optim as optim

class IrisNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=3):
        super(IrisNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.output(x)
        return x