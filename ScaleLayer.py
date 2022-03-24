import numpy as np
import torch
import torch.nn as nn


class ScaleLayer(nn.Module):
    
   def __init__(self, size, init_value=1.0, dim=4):
       super(ScaleLayer, self).__init__()
       self.scale = nn.Parameter(init_value*torch.ones(size))
       self.dim = dim

   def forward(self, input):
       #print 'input size : '+ str(input.size())
       #print 'scale size : '+ str(self.scale.unsqueeze(0).unsqueeze(2).unsqueeze(3).data.size())
        if self.dim==4:
            return torch.mul(input, self.scale.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(input))  
        elif self.dim==2:
            return torch.mul(input, self.scale.unsqueeze(0).expand_as(input)) 
