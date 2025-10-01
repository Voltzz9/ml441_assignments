# imports
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=3, use_mse=False):
        super(NeuralNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.hidden_act = nn.Sigmoid()
        self.output_act = nn.Sigmoid()
        self.use_mse = use_mse
        
    def forward(self, x):
        x = self.hidden_act(self.hidden(x))
        x = self.output(x)
        if self.use_mse:
            # Apply sigmoid to output for MSE (outputs in [0,1] range)
            x = self.output_act(x)
        # Return raw logits for CrossEntropyLoss, sigmoid outputs for MSE
        return x

class RegressionNet(nn.Module):
    """Neural network specifically designed for regression tasks"""
    def __init__(self, input_size=1, hidden_size=128, output_size=1):
        super(RegressionNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.hidden_act = nn.Sigmoid()
        # No output activation - allows full range of outputs
        
    def forward(self, x):
        x = self.hidden_act(self.hidden(x))
        x = self.output(x)  # No activation on output for regression
        return x