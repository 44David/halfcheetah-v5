import torch.nn as nn

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(17, 64)
        self.hidden = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output_layer(x)
        
        return x