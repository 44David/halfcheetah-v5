import torch.nn as nn
import torch

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(17, 128)
        self.hidden = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 6)
        self.log_std = nn.Parameter(torch.ones(6) * -2.0)   
        
        
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden(x)
        x = self.output_layer(x)
        
        return x
    
    

        
    