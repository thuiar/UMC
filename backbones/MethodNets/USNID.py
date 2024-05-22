import torch.nn as nn

class USNIDModel(nn.Module):

    def __init__(self, args, backbone):

        super(USNIDModel, self).__init__()
        self.backbone = backbone
        activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}
        self.dense = nn.Linear(args.feat_dim, args.feat_dim)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.classifier = nn.Linear(args.feat_dim, args.num_labels)
        self.mlp_head = nn.Linear(args.feat_dim, args.num_labels)

    def forward(self, text, video, audio, feature_ext=False):
        features = self.backbone(text, video, audio)
        features = self.dense(features)
        pooled_output = self.activation(features)  
        pooled_output = self.dropout(features)
        mlp_outputs = self.mlp_head(pooled_output)
        
        if feature_ext:
            return features, mlp_outputs
        else:
            logits = self.classifier(pooled_output)
            return mlp_outputs, logits
