import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}

class SCCLModel(nn.Module):
    
    def __init__(self, args, backbone):
        super(SCCLModel, self).__init__()
        self.backbone = backbone
        self.contrast_head = None
        self.cluster_centers = None

    def init_model(self, cluster_centers=None, alpha=1.0):

        # self.emb_size = self.bert.config.hidden_size
        self.emb_size = 768
        self.alpha = alpha
        
        # Instance-CL head
        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, 128))
        
        # Clustering head
        initial_cluster_centers = torch.tensor(
            cluster_centers, dtype=torch.float, requires_grad=True)
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, text, video, audio):
        return self.backbone.forward(text, video, audio)

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def contrast_logits(self, embd1, embd2=None):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        if embd2 != None:
            feat2 = F.normalize(self.contrast_head(embd2), dim=1)
            return feat1, feat2
        else: 
            return feat1

  