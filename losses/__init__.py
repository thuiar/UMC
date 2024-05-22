from torch import nn 
from .SupConLoss import SupConLoss

loss_map = {
                'CrossEntropyLoss': nn.CrossEntropyLoss(), 
                'SupConLoss': SupConLoss(),
                'MSELoss': nn.MSELoss(),
            }
