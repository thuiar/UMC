import torch
import torch.nn as nn
import torch.nn.functional as F

class CCModel(nn.Module):

    def __init__(self, args, backbone):

        super(CCModel, self).__init__()
        self.backbone = backbone

        self.instance_projector = nn.Sequential(
            nn.Linear(args.feat_dim, args.feat_dim),
            nn.ReLU(),
            nn.Linear(args.feat_dim, args.feat_dim),
        )

        self.cluster_projector = nn.Sequential(
            nn.Linear(args.feat_dim, args.feat_dim),
            nn.ReLU(),
            nn.Linear(args.feat_dim, args.num_labels),
            nn.Softmax(dim=1),
        )
        
    def forward(self, text, video, audio):
        
        features = self.backbone(text, video, audio)
        return features
    
    def get_features(self, h_i, h_j):
        
        z_i = F.normalize(self.instance_projector(h_i), dim=1)
        z_j = F.normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
     
        c = self.cluster_projector(x)
        c = torch.argmax(c, dim=1)
        return c
