import torch.nn as nn

class MCNModel(nn.Module):

    def __init__(self, args, backbone):

        super(MCNModel, self).__init__()
        self.backbone = backbone
    
    def forward(self, text, video, audio, mode='train'):

        return self.backbone(text, video, audio, mode)
    